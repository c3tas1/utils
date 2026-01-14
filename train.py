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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Add tqdm for progress bars
import warnings

# Import from model_utils
from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

# Suppress DDP warnings to clean up logs
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

def safe_tensor_to_img(img_tensor):
    """
    Bulletproof conversion to uint8 to prevent 'integer out of bounds' errors.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    
    img = img_tensor * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).type(torch.uint8)
    
    return img.permute(1, 2, 0).cpu().numpy()

def save_visual_check(model, dataset, device, epoch, output_dir):
    model.eval()
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_mil_pad, shuffle=True)
    try:
        images, labels, mask, pids = next(iter(loader))
    except StopIteration:
        return
        
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
    
    # Use autocast for inference too (matches training dtype)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            bag_logits, inst_logits, attn = model(images, mask)
            preds = torch.argmax(bag_logits, dim=1)
        
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5*len(images)))
    if len(images) == 1: axes = [axes]
    
    classes = dataset.classes
    
    for i in range(len(images)):
        num_pills = int(mask[i].sum().item())
        curr_imgs_tensor = images[i][:num_pills]
        
        display_imgs = []
        for p_idx in range(curr_imgs_tensor.shape[0]):
            img_np = safe_tensor_to_img(curr_imgs_tensor[p_idx])
            display_imgs.append(img_np)
            
        curr_attn = attn[i][:num_pills].float().cpu().view(-1).numpy()
        pred_lbl = classes[preds[i]]
        true_lbl = classes[labels[i]]
        
        grid_img = np.concatenate(display_imgs, axis=1)
        
        axes[i].imshow(grid_img)
        title = f"GT: {true_lbl} | Pred: {pred_lbl} | Attn: {np.round(curr_attn, 2)}"
        axes[i].set_title(title, color='green' if pred_lbl==true_lbl else 'red')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch}_visual_check.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--accum_steps", type=int, default=16) 
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # --- Transforms ---
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PrescriptionDataset(os.path.join(args.data_dir, 'train'), transform=train_tfm)
    val_ds = PrescriptionDataset(os.path.join(args.data_dir, 'valid'), transform=val_tfm)
    
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              collate_fn=collate_mil_pad, num_workers=4, pin_memory=True)
    
    model = GatedAttentionMIL(num_classes=len(train_ds.classes)).to(device)
    
    # CLEANUP: Removed find_unused_parameters=True to stop the warning spam. 
    # Your model graph is fully connected, so this is safe and faster.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        run_bag_corr = 0
        run_bag_total = 0
        run_inst_corr = 0
        run_inst_total = 0
        
        optimizer.zero_grad()
        
        # ADDED: TQDM Progress Bar (Only on Rank 0 to avoid clutter)
        if local_rank == 0:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        else:
            pbar = enumerate(train_loader)
            
        for i, (images, labels, mask, _) in pbar:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            
            with torch.cuda.amp.autocast():
                bag_logits, inst_logits, _ = model(images, mask)
                
                loss_bag = criterion(bag_logits, labels)
                
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
            
            # --- Metrics Calculation (Restored Pill Acc) ---
            # Bag Metrics
            bag_preds = torch.argmax(bag_logits, dim=1)
            run_bag_corr += (bag_preds == labels).sum().item()
            run_bag_total += labels.size(0)
            
            # Pill Metrics
            inst_preds = torch.argmax(flat_inst_logits, dim=1)
            valid_inst_preds = inst_preds[flat_mask==1]
            valid_inst_labels = flat_labels[flat_mask==1]
            run_inst_corr += (valid_inst_preds == valid_inst_labels).sum().item()
            run_inst_total += valid_inst_labels.size(0)

            # Update Progress Bar (Rank 0)
            if local_rank == 0 and i % 10 == 0:
                current_loss = total_loss.item() * args.accum_steps
                bag_acc = run_bag_corr/run_bag_total
                pill_acc = run_inst_corr/run_inst_total if run_inst_total > 0 else 0
                
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "Bag Acc": f"{bag_acc:.2%}",
                    "Pill Acc": f"{pill_acc:.2%}"
                })

        scheduler.step()
        
        if local_rank == 0:
            print(f"--> Saving Checkpoint and Visuals for Epoch {epoch}")
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            save_visual_check(model, val_ds, device, epoch, ".")

    cleanup_ddp()

if __name__ == "__main__":
    main()
