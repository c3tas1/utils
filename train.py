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

from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

def save_visual_check(model, dataset, device, epoch, output_dir):
    """Saves a visual grid of one batch showing attention and predictions"""
    model.eval()
    # Create a small temp loader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_mil_pad, shuffle=True)
    images, labels, mask, pids = next(iter(loader))
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
    
    with torch.no_grad():
        bag_logits, inst_logits, attn = model(images, mask)
        preds = torch.argmax(bag_logits, dim=1)
        
    # Plotting Logic
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5*len(images)))
    if len(images) == 1: axes = [axes]
    
    classes = dataset.classes
    
    for i in range(len(images)):
        # Unpack pills
        num_pills = int(mask[i].sum().item())
        curr_imgs = images[i][:num_pills].cpu()
        curr_attn = attn[i][:num_pills].cpu().view(-1).numpy()
        pred_lbl = classes[preds[i]]
        true_lbl = classes[labels[i]]
        
        # Create a grid for this prescription
        # We concat images horizontally for visualization
        grid_img = np.concatenate([np.transpose(img.numpy(), (1, 2, 0)) for img in curr_imgs], axis=1)
        
        # Normalize for display
        grid_img = (grid_img - grid_img.min()) / (grid_img.max() - grid_img.min())
        
        axes[i].imshow(grid_img)
        title = f"GT: {true_lbl} | Pred: {pred_lbl} | Attn Weights: {np.round(curr_attn, 2)}"
        axes[i].set_title(title, color='green' if pred_lbl==true_lbl else 'red')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch}_visual_check.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16) # Batch size per GPU
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # --- Transforms (Heavy augmentation for lighting issues) ---
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Data Loading ---
    train_ds = PrescriptionDataset(os.path.join(args.data_dir, 'train'), transform=train_tfm)
    val_ds = PrescriptionDataset(os.path.join(args.data_dir, 'valid'), transform=val_tfm)
    
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              collate_fn=collate_mil_pad, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_mil_pad, num_workers=4)

    # --- Model Setup ---
    # Note: 1228 classes as per prompt
    model = GatedAttentionMIL(num_classes=len(train_ds.classes)).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        run_bag_corr = 0
        run_bag_total = 0
        run_inst_corr = 0
        run_inst_total = 0
        
        for i, (images, labels, mask, _) in enumerate(train_loader):
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            bag_logits, inst_logits, _ = model(images, mask)
            
            # --- 1. Bag (Prescription) Loss ---
            loss_bag = criterion(bag_logits, labels)
            
            # --- 2. Instance (Pill) Loss ---
            # We expand labels to match pills: (B) -> (B, M)
            B, M, _ = inst_logits.shape
            expanded_labels = labels.unsqueeze(1).repeat(1, M) # (B, M)
            
            # Flatten for loss calculation
            # Only calculate loss for real pills (using mask)
            flat_inst_logits = inst_logits.view(B*M, -1)
            flat_labels = expanded_labels.view(-1)
            flat_mask = mask.view(-1)
            
            loss_inst = criterion(flat_inst_logits[flat_mask==1], flat_labels[flat_mask==1])
            
            # Combined Loss (Weighting bag loss higher as it is the primary goal)
            total_loss = loss_bag + (0.5 * loss_inst)
            
            total_loss.backward()
            optimizer.step()
            
            # --- Metrics Calculation ---
            # Bag Accuracy
            bag_preds = torch.argmax(bag_logits, dim=1)
            run_bag_corr += (bag_preds == labels).sum().item()
            run_bag_total += labels.size(0)
            
            # Instance Accuracy
            inst_preds = torch.argmax(flat_inst_logits, dim=1)
            # Filter by mask for accuracy
            valid_inst_preds = inst_preds[flat_mask==1]
            valid_inst_labels = flat_labels[flat_mask==1]
            run_inst_corr += (valid_inst_preds == valid_inst_labels).sum().item()
            run_inst_total += valid_inst_labels.size(0)

            if local_rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}] "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Bag Acc: {run_bag_corr/run_bag_total:.2%} | "
                      f"Pill Acc: {run_inst_corr/run_inst_total:.2%}")

        scheduler.step()
        
        # --- Validation & Viz (Rank 0 only) ---
        if local_rank == 0:
            print(f"--> Saving Checkpoint and Visuals for Epoch {epoch}")
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            save_visual_check(model, val_ds, device, epoch, ".")

    cleanup_ddp()

if __name__ == "__main__":
    main()
