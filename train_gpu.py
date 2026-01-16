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
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # --- FIX: Stronger Augmentation ---
        # Prevents model from memorizing specific "glare" patterns
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Added Vertical Flip
            transforms.RandomRotation(180),  # Full rotation
            # Stronger Color Jitter
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

def save_visual_check(model, dataset, device, epoch, output_dir):
    model.eval()
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_mil_pad, shuffle=True, num_workers=0)
    try:
        images, labels, mask, pids = next(iter(loader))
    except StopIteration:
        return
        
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    
    B, M, C, H, W = images.shape
    images_norm = images.view(B*M, C, H, W)
    images_norm = norm(images_norm)
    images_norm = images_norm.view(B, M, C, H, W)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            bag_logits, inst_logits, attn = model(images_norm, mask)
            preds = torch.argmax(bag_logits, dim=1)
        
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5*len(images)))
    if len(images) == 1: axes = [axes]
    
    classes = dataset.classes
    for i in range(len(images)):
        num_pills = int(mask[i].sum().item())
        curr_imgs_tensor = images[i][:num_pills] 
        display_imgs = []
        for p_idx in range(curr_imgs_tensor.shape[0]):
            img_t = curr_imgs_tensor[p_idx]
            img_t = torch.clamp(img_t, 0.0, 1.0)
            img_np = (img_t * 255.0).type(torch.uint8).permute(1, 2, 0).cpu().numpy()
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

def validate(model, val_loader, device, gpu_aug):
    model.eval()
    val_bag_corr = 0
    val_bag_total = 0
    val_inst_corr = 0
    val_inst_total = 0
    
    with torch.no_grad():
        for images, labels, mask, _ in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            images = gpu_aug(images, training=False)
            
            with torch.cuda.amp.autocast():
                bag_logits, inst_logits, _ = model(images, mask)
                
            bag_preds = torch.argmax(bag_logits, dim=1)
            val_bag_corr += (bag_preds == labels).sum().item()
            val_bag_total += labels.size(0)
            
            B, M, _ = inst_logits.shape
            flat_inst_logits = inst_logits.view(B*M, -1)
            flat_mask = mask.view(-1)
            flat_labels = labels.unsqueeze(1).repeat(1, M).view(-1)
            
            inst_preds = torch.argmax(flat_inst_logits, dim=1)
            valid_preds = inst_preds[flat_mask==1]
            valid_labels = flat_labels[flat_mask==1]
            
            if valid_labels.size(0) > 0:
                val_inst_corr += (valid_preds == valid_labels).sum().item()
                val_inst_total += valid_labels.size(0)
                
    return val_bag_corr/val_bag_total, val_inst_corr/val_inst_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--accum_steps", type=int, default=8) 
    parser.add_argument("--workers", type=int, default=12) 
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    cpu_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])
    gpu_aug = GPUAugmentation().to(device)

    train_ds = PrescriptionDataset(os.path.join(args.data_dir, 'train'), transform=cpu_tfm, cache_ram=False)
    val_ds = PrescriptionDataset(os.path.join(args.data_dir, 'valid'), transform=cpu_tfm, cache_ram=False)
    
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              collate_fn=collate_mil_pad, num_workers=args.workers, 
                              pin_memory=True, prefetch_factor=4, persistent_workers=True) 
                              
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_mil_pad, 
                            num_workers=4, shuffle=False)

    model = GatedAttentionMIL(num_classes=len(train_ds.classes)).to(device)
    
    start_epoch = args.start_epoch
    loaded_opt_state = None
    
    # --- RESUME LOGIC ---
    if args.resume and os.path.isfile(args.resume):
        if local_rank == 0:
            print(f"--> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # We can safely load weights even with new Dropout layers
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Strict=False just in case, but usually fine
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            if 'optimizer_state_dict' in checkpoint: loaded_opt_state = checkpoint['optimizer_state_dict']
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # --- FIX: Stronger L2 Regularization (weight_decay=1e-3) ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    if loaded_opt_state: 
        optimizer.load_state_dict(loaded_opt_state)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        run_bag_corr = 0
        run_bag_total = 0
        run_inst_corr = 0
        run_inst_total = 0
        
        optimizer.zero_grad()
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        else:
            pbar = train_loader
            
        for i, (images, labels, mask, _) in enumerate(pbar):
            images, labels, mask = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            images = gpu_aug(images, training=True)
            
            with torch.cuda.amp.autocast():
                bag_logits, inst_logits, _ = model(images, mask)
                loss_bag = criterion(bag_logits, labels)
                B, M, _ = inst_logits.shape
                flat_inst_logits = inst_logits.view(B*M, -1)
                flat_labels = labels.unsqueeze(1).repeat(1, M).view(-1)
                flat_mask = mask.view(-1)
                loss_inst = criterion(flat_inst_logits[flat_mask==1], flat_labels[flat_mask==1])
                total_loss = (loss_bag + (0.5 * loss_inst)) / args.accum_steps
            
            scaler.scale(total_loss).backward()
            
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if i % 10 == 0:
                bag_preds = torch.argmax(bag_logits, dim=1)
                run_bag_corr += (bag_preds == labels).sum().item()
                run_bag_total += labels.size(0)
                
                inst_preds = torch.argmax(flat_inst_logits, dim=1)
                valid_preds = inst_preds[flat_mask==1]
                valid_labels = flat_labels[flat_mask==1]
                if valid_labels.size(0) > 0:
                    run_inst_corr += (valid_preds == valid_labels).sum().item()
                    run_inst_total += valid_labels.size(0)

                if local_rank == 0 and i % 50 == 0:
                     pill_acc = run_inst_corr/run_inst_total if run_inst_total > 0 else 0
                     pbar.set_postfix({
                        "Loss": f"{total_loss.item() * args.accum_steps:.4f}",
                        "Train Bag": f"{run_bag_corr/run_bag_total:.2%}",
                        "Train Pill": f"{pill_acc:.2%}"
                    })

        scheduler.step()
        
        val_bag_acc, val_inst_acc = validate(model, val_loader, device, gpu_aug)
        
        if local_rank == 0:
            print(f"\n--> Epoch {epoch} COMPLETE")
            print(f"    Train Bag Acc: {run_bag_corr/run_bag_total:.2%} | Train Pill Acc: {run_inst_corr/run_inst_total:.2%}")
            print(f"    VALID Bag Acc: {val_bag_acc:.2%} | VALID Pill Acc: {val_inst_acc:.2%}")
            print("-" * 60)
            
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, f"checkpoint_epoch_{epoch}.pth")
            save_visual_check(model, val_ds, device, epoch, ".")

    cleanup_ddp()

if __name__ == "__main__":
    main()
