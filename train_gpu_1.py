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
from tqdm import tqdm
import warnings

# Import your existing utilities
from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()

def calculate_accuracy(logits, labels, mask=None):
    """Helper to calculate accuracy for a batch"""
    # --- FIX START ---
    # Check if this is Bag Logits (2D) or Instance Logits (3D)
    if logits.dim() == 3:
        # Instance Level: [Batch, Pills, Classes] -> argmax on dim 2
        preds = torch.argmax(logits, dim=2)
    else:
        # Bag Level: [Batch, Classes] -> argmax on dim 1
        preds = torch.argmax(logits, dim=1)
    # --- FIX END ---
    
    if mask is None:
        # Bag Level
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    else:
        # Instance Level
        flat_preds = preds.view(-1)
        
        # Expand labels to match the number of pills
        num_pills = logits.shape[1]
        flat_labels = labels.unsqueeze(1).repeat(1, num_pills).view(-1)
        flat_mask = mask.view(-1)
        
        valid_preds = flat_preds[flat_mask == 1]
        valid_labels = flat_labels[flat_mask == 1]
        
        correct = (valid_preds == valid_labels).sum().item()
        total = valid_labels.size(0)
        
    return correct, total

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    
    correct_bags = 0
    total_bags = 0
    correct_pills = 0
    total_pills = 0
    
    with torch.no_grad():
        for imgs, labels, mask, _ in val_loader:
            labels = labels.to(device)
            mask = mask.to(device)
            
            bag_logits, inst_logits, _ = model(imgs, mask)
            
            # 1. Bag Metrics
            loss = criterion(bag_logits, labels)
            val_loss += loss.item()
            
            b_corr, b_tot = calculate_accuracy(bag_logits, labels)
            correct_bags += b_corr
            total_bags += b_tot
            
            # 2. Pill Metrics
            i_corr, i_tot = calculate_accuracy(inst_logits, labels, mask)
            correct_pills += i_corr
            total_pills += i_tot
            
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    bag_acc = correct_bags / total_bags if total_bags > 0 else 0
    pill_acc = correct_pills / total_pills if total_pills > 0 else 0
    
    return avg_loss, bag_acc, pill_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TRAIN folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to VALID folder")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to .xml model")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to .bin model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # --- DATA SETUP ---
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Train Loader
    train_ds = PrescriptionDataset(args.data_dir, transform=tfm)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_mil_pad, num_workers=4)

    # Val Loader
    val_ds = PrescriptionDataset(args.val_dir, transform=tfm)
    # Ensure val classes match train classes exactly
    val_ds.class_to_idx = train_ds.class_to_idx 
    val_ds.classes = train_ds.classes
    
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_mil_pad, num_workers=4)

    # --- MODEL INIT ---
    if local_rank == 0:
        print(f"--> Initializing Hybrid Model... (Train: {len(train_ds)} bags, Val: {len(val_ds)} bags)")

    model = GatedAttentionMIL(num_classes=len(train_ds.classes), xml_path=args.xml_path, bin_path=args.bin_path)
    model.attention_V.to(device)
    model.attention_U.to(device)
    model.attention_weights.to(device)
    model.bag_classifier.to(device)
    model.instance_classifier.to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        start_epoch = chk['epoch'] + 1

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        r_bag_corr = 0
        r_bag_tot = 0
        r_pill_corr = 0
        r_pill_tot = 0
        
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader
            
        for imgs, labels, mask, _ in pbar:
            labels = labels.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            bag_logits, inst_logits, _ = model(imgs, mask)
            
            # Loss
            loss_bag = criterion(bag_logits, labels)
            
            # Instance Loss (Approximation)
            B, M, _ = inst_logits.shape
            flat_inst = inst_logits.view(B*M, -1)
            flat_labels = labels.unsqueeze(1).repeat(1, M).view(-1)
            flat_mask = mask.view(-1)
            loss_inst = criterion(flat_inst[flat_mask==1], flat_labels[flat_mask==1])
            
            total_loss = loss_bag + (0.5 * loss_inst)
            total_loss.backward()
            optimizer.step()
            
            # --- RUNNING METRICS ---
            b_c, b_t = calculate_accuracy(bag_logits, labels)
            i_c, i_t = calculate_accuracy(inst_logits, labels, mask)
            
            running_loss += total_loss.item()
            r_bag_corr += b_c
            r_bag_tot += b_t
            r_pill_corr += i_c
            r_pill_tot += i_t
            
            if local_rank == 0 and isinstance(pbar, tqdm):
                curr_bag_acc = r_bag_corr / r_bag_tot if r_bag_tot > 0 else 0
                curr_pill_acc = r_pill_corr / r_pill_tot if r_pill_tot > 0 else 0
                pbar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}", 
                    "RxAcc": f"{curr_bag_acc:.1%}", 
                    "PillAcc": f"{curr_pill_acc:.1%}"
                })

        # --- VALIDATION ---
        # Note: We run this on all ranks, but only print Rank 0
        val_loss, val_bag_acc, val_pill_acc = validate(model, val_loader, device, criterion)
        
        if local_rank == 0:
            print("-" * 50)
            print(f"EPOCH {epoch+1} SUMMARY")
            print("-" * 50)
            print(f"{'METRIC':<20} | {'TRAIN':<10} | {'VALIDATION':<10}")
            print(f"{'Script (Bag) Acc':<20} | {r_bag_corr/r_bag_tot:.2%}     | {val_bag_acc:.2%}")
            print(f"{'Pill (Inst) Acc':<20}  | {r_pill_corr/r_pill_tot:.2%}     | {val_pill_acc:.2%}")
            print(f"{'Loss':<20}          | {running_loss/len(train_loader):.4f}     | {val_loss:.4f}")
            print("-" * 50)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch+1}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()
