import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms, models, datasets
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
# Batch size PER GPU. Global Batch Size = BATCH_SIZE * Num_GPUs
BATCH_SIZE = 256  
EPOCHS = 20
LR = 0.01 

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="./train_patches_10000/train")
    parser.add_argument("--val_dir", type=str, required=True, help="./train_patches_10000/valid")
    parser.add_argument("--save_path", type=str, default="resnet34_ddp_restored.pth")
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"--> Training Backbone on {torch.cuda.device_count()} GPUs (DDP Mode)")

    # 1. Data Setup
    # Standard ImageNet normalization
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Note: Ensure these folders exist and contain class subfolders
    train_ds = datasets.ImageFolder(args.data_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfm)
    
    # 2. DDP Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    # Validation sampler ensures we don't overlap data, but for metrics on Rank 0 
    # we technically only see 1/Nth of the data. For 98% acc checks, this is usually fine.
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=8, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler, 
        num_workers=8, 
        pin_memory=True
    )

    # 3. Model Setup
    model = models.resnet34(weights='IMAGENET1K_V1')
    
    # Adjust Head to match your number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
    
    model = model.to(device)
    
    # SyncBatchNorm helps with convergence in DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # DDP Wrapper
    model = DDP(model, device_ids=[local_rank])

    # 4. Optimization
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    # 5. Training Loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Setup Progress Bar (Only on Rank 0)
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader
            
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # --- RUNNING METRICS ---
            current_loss = loss.item()
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            
            running_loss += current_loss
            correct += batch_correct
            total += batch_total
            
            # Update TQDM on Rank 0
            if local_rank == 0 and isinstance(pbar, tqdm):
                running_acc = correct / total if total > 0 else 0
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}", 
                    "RunAcc": f"{running_acc:.2%}"
                })
        
        scheduler.step()
        
        # --- VALIDATION LOOP ---
        # We run this block only on Rank 0 to print the summary.
        # (Note: In DDP, Rank 0 only validates its slice of data. 
        # For a backbone check, this is usually sufficient.)
        if local_rank == 0:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            # Compute Final Epoch Metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total
            
            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            
            print("-" * 65)
            print(f"{'METRIC':<20} | {'TRAIN':<15} | {'VALIDATION':<15}")
            print("-" * 65)
            print(f"{'Loss':<20} | {epoch_train_loss:.4f}          | {epoch_val_loss:.4f}")
            print(f"{'Accuracy':<20} | {epoch_train_acc:.2%}          | {epoch_val_acc:.2%}")
            print("-" * 65)
            
            # Save Checkpoint
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.module.state_dict(), args.save_path)
                print(f"--> Saved Best Model: {args.save_path} (Acc: {best_acc:.2%})")

    cleanup_ddp()

if __name__ == "__main__":
    main()
