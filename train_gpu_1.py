import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce
from torchvision import transforms
from tqdm import tqdm
import warnings

# Import the hybrid components
from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

warnings.filterwarnings("ignore")
# Optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data (Class folders)")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to ResNet34.xml")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to ResNet34.bin")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # --- DATA PREP ---
    # Standard normalization for ResNet
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    train_ds = PrescriptionDataset(args.data_dir, transform=tfm)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=collate_mil_pad, 
        num_workers=4, # Keep low/moderate as OpenVINO uses CPU too
        pin_memory=False # False because we process on CPU first
    )

    # --- MODEL INIT ---
    print(f"--> Initializing Hybrid Model on Rank {local_rank}...")
    model = GatedAttentionMIL(
        num_classes=len(train_ds.classes),
        xml_path=args.xml_path,
        bin_path=args.bin_path
    )
    
    # Move TRAINABLE parts to GPU
    model.attention_V.to(device)
    model.attention_U.to(device)
    model.attention_weights.to(device)
    model.bag_classifier.to(device)
    model.instance_classifier.to(device)

    # DDP Wrapper
    # Note: We set find_unused_parameters=True because the backbone is technically "unused" 
    # by the PyTorch optimizer (it's frozen/OpenVINO).
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Resume Logic
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"--> Loading checkpoint {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        start_epoch = chk['epoch'] + 1

    # --- TRAINING LOOP ---
    print("--> Starting Training...")
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader
            
        for imgs, labels, mask, _ in pbar:
            # Labels/Mask to GPU immediately
            labels = labels.to(device)
            mask = mask.to(device)
            # Images stay on CPU for OpenVINO first, then features go to GPU inside forward()
            
            optimizer.zero_grad()
            
            # Forward
            bag_logits, inst_logits, _ = model(imgs, mask)
            
            # Loss Calculation
            loss_bag = criterion(bag_logits, labels)
            
            # Instance Loss (Max Pooling approximation)
            B, M, _ = inst_logits.shape
            flat_inst = inst_logits.view(B*M, -1)
            flat_labels = labels.unsqueeze(1).repeat(1, M).view(-1)
            flat_mask = mask.view(-1)
            loss_inst = criterion(flat_inst[flat_mask==1], flat_labels[flat_mask==1])
            
            total_loss = loss_bag + (0.5 * loss_inst)
            
            total_loss.backward()
            optimizer.step()
            
            if local_rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        # Save Checkpoint
        if local_rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': train_ds.classes # Save classes for inference convenience
            }, f"checkpoint_epoch_{epoch}.pth")
            
    cleanup_ddp()

if __name__ == "__main__":
    main()
