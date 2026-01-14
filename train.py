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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Un-normalizes a tensor image."""
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def save_visual_check(model, dataset, device, epoch, output_dir):
    """
    Saves a visual grid.
    FIXED: Uses UnNormalize and Clamp to prevent OverflowError on negative pixels.
    """
    model.eval()
    
    # Define UnNormalizer with same stats as training
    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create small temp loader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_mil_pad, shuffle=True)
    images, labels, mask, pids = next(iter(loader))
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
    
    with torch.no_grad():
        bag_logits, inst_logits, attn = model(images, mask)
        preds = torch.argmax(bag_logits, dim=1)
        
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5*len(images)))
    if len(images) == 1: axes = [axes]
    
    classes = dataset.classes
    
    for i in range(len(images)):
        num_pills = int(mask[i].sum().item())
        
        # Clone to avoid modifying original tensor
        curr_imgs_tensor = images[i][:num_pills].detach().cpu().clone()
        
        display_imgs = []
        for p_idx in range(curr_imgs_tensor.shape[0]):
            # Un-normalize
            img_unorm = unorm(curr_imgs_tensor[p_idx]) 
            # Clamp to [0,1] to strictly avoid negative numbers or >1
            img_unorm = torch.clamp(img_unorm, 0, 1)
            # Convert to Numpy (H, W, C)
            img_np = img_unorm.permute(1, 2, 0).numpy()
            display_imgs.append(img_np)

        curr_attn = attn[i][:num_pills].cpu().view(-1).numpy()
        pred_lbl = classes[preds[i]]
        true_lbl = classes[labels[i]]
        
        # Concatenate horizontally
        grid_img = np.concatenate(display_imgs, axis=1)
        
        axes[i].imshow(grid_img)
        title = f"GT: {true_lbl} | Pred: {pred_lbl} | Attn Weights: {np.round(curr_attn, 2)}"
        axes[i].set_title(title, color='green' if pred_lbl==true_lbl else 'red')
        axes[i].axis('off')
        
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/epoch_{epoch}_visual_check.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # --- Transforms ---
    # Strong augmentation to handle your lighting/noise issues
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
    # Validation loader only needed on Rank 0 for visuals usually, but good to have everywhere
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_mil_pad, num_workers=4)

    # --- Model Setup ---
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
            
            # --- Bag Loss (Main Goal) ---
            loss_bag = criterion(bag_logits, labels)
            
            # --- Instance Loss (Auxiliary Goal) ---
            B, M, _ = inst_logits.shape
            expanded_labels = labels.unsqueeze(1).repeat(1, M)
            flat_inst_logits = inst_logits.view(B*M, -1)
            flat_labels = expanded_labels.view(-1)
            flat_mask = mask.view(-1)
            
            # Only calc loss on real pills (mask=1)
            loss_inst = criterion(flat_inst_logits[flat_mask==1], flat_labels[flat_mask==1])
            
            total_loss = loss_bag + (0.5 * loss_inst)
            
            total_loss.backward()
            optimizer.step()
            
            # --- Metrics ---
            bag_preds = torch.argmax(bag_logits, dim=1)
            run_bag_corr += (bag_preds == labels).sum().item()
            run_bag_total += labels.size(0)
            
            inst_preds = torch.argmax(flat_inst_logits, dim=1)
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
        
        # --- Save Checkpoints & Visuals (Rank 0 only) ---
        if local_rank == 0:
            print(f"--> Saving Checkpoint and Visuals for Epoch {epoch}")
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            try:
                save_visual_check(model, val_ds, device, epoch, "./visual_logs")
            except Exception as e:
                print(f"Visual check failed: {e}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
