import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import warnings

from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

warnings.filterwarnings("ignore")

# --- SPEED HACK 1: ENABLE TF32 ON A100 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.val_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, training=True):
        B, M, C, H, W = x.shape
        x = x.view(B * M, C, H, W) 
        if training:
            x = self.transforms(x)
        else:
            x = self.val_norm(x)
        return x.view(B, M, C, H, W)

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    # Batch size 4 is usually safe with Gradient Checkpointing on A100
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--accum_steps", type=int, default=8) # Effective Batch = 4*8*8GPUs = 256
    parser.add_argument("--workers", type=int, default=12) # Increased workers
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # Lightweight CPU transforms (Resize only)
    cpu_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])
    
    gpu_aug = GPUAugmentation().to(device)

    # --- SPEED HACK 2: RAM CACHING ---
    # If your DGX has >1TB RAM and your dataset fits (check size!), set cache_ram=True
    # WARNING: Watch your RAM usage. If it crashes, set back to False.
    train_ds = PrescriptionDataset(os.path.join(args.data_dir, 'train'), transform=cpu_tfm, cache_ram=False)
    val_ds = PrescriptionDataset(os.path.join(args.data_dir, 'valid'), transform=cpu_tfm, cache_ram=False)
    
    train_sampler = DistributedSampler(train_ds)
    
    # --- SPEED HACK 3: LOADER OPTIMIZATION ---
    # persistent_workers=True keeps the RAM loaded. 
    # prefetch_factor=4 ensures the next batch is ready before GPU asks.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              collate_fn=collate_mil_pad, 
                              num_workers=args.workers, 
                              pin_memory=True, 
                              prefetch_factor=4, 
                              persistent_workers=True) 

    model = GatedAttentionMIL(num_classes=len(train_ds.classes)).to(device)
    
    # --- SPEED HACK 4: TORCH COMPILE ---
    # Fuses kernels for massive A100 speedup
    print("Compiling model... (This takes a minute at start)")
    model = torch.compile(model)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        run_bag_corr = 0
        run_bag_total = 0
        
        optimizer.zero_grad()
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader
            
        for i, (images, labels, mask, _) in enumerate(pbar):
            images, labels, mask = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            
            images = gpu_aug(images, training=True)
            
            with torch.cuda.amp.autocast():
                bag_logits, inst_logits, _ = model(images, mask)
                loss_bag = criterion(bag_logits, labels)
                
                # Simplified loss calculation for speed
                B, M, _ = inst_logits.shape
                expanded_labels = labels.unsqueeze(1).repeat(1, M)
                flat_inst_logits = inst_logits.view(B*M, -1)
                flat_labels = expanded_labels.view(-1)
                flat_mask = mask.view(-1)
                
                loss_inst = criterion(flat_inst_logits[flat_mask==1], flat_labels[flat_mask==1])
                total_loss = (loss_bag + (0.5 * loss_inst)) / args.accum_steps
            
            scaler.scale(total_loss).backward()
            
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Metrics (Calculated less frequently to save CPU time)
            if i % 5 == 0:
                bag_preds = torch.argmax(bag_logits, dim=1)
                run_bag_corr += (bag_preds == labels).sum().item()
                run_bag_total += labels.size(0)

                if local_rank == 0 and i % 50 == 0:
                     pbar.set_postfix({
                        "Loss": f"{total_loss.item() * args.accum_steps:.4f}",
                        "Bag Acc": f"{run_bag_corr/run_bag_total:.2%}",
                    })

        scheduler.step()
        
        if local_rank == 0:
            print(f"--> Saving Checkpoint Epoch {epoch}")
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()
