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

# Import from model_utils
from model_utils import PrescriptionDataset, collate_mil_pad, GatedAttentionMIL

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

def safe_tensor_to_img(img_tensor):
    """
    Bulletproof conversion from Tensor to Numpy Image.
    1. Un-normalizes on GPU.
    2. Clamps to [0, 1].
    3. Scales to [0, 255].
    4. Casts to uint8 ON GPU to avoid Python overflow errors.
    """
    # ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    
    # Un-normalize
    img = img_tensor * std + mean
    
    # Strict Clamp (Forces all values to be valid 0.0 - 1.0)
    img = torch.clamp(img, 0.0, 1.0)
    
    # Scale to 255 and convert to ByteTensor (uint8)
    img = (img * 255.0).type(torch.uint8)
    
    # Move to CPU and format for Matplotlib (H, W, C)
    return img.permute(1, 2, 0).cpu().numpy()

def save_visual_check(model, dataset, device, epoch, output_dir):
    model.eval()
    
    # Create small batch
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_mil_pad, shuffle=True)
    try:
        images, labels, mask, pids = next(iter(loader))
    except StopIteration:
        return
        
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
    
    with torch.no_grad():
        bag_logits, inst_logits, attn = model(images, mask)
        preds = torch.argmax(bag_logits, dim=1)
        
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5*len(images)))
    if len(images) == 1: axes = [axes]
    
    classes = dataset.classes
    
    for i in range(len(images)):
        num_pills = int(mask[i].sum().item())
        
        # Get raw tensor for pills in this bag
        curr_imgs_tensor = images[i][:num_pills] # (Num_Pills, C, H, W)
        
        display_imgs = []
        for p_idx in range(curr_imgs_tensor.shape[0]):
            # Use safe converter
            img_np = safe_tensor_to_img(curr_imgs_tensor[p_idx])
            display_imgs.append(img_np)
            
        curr_attn = attn[i][:num_pills].cpu().view(-1).numpy()
        pred_lbl = classes[preds[i]]
        true_lbl = classes[labels[i]]
        
        # Concatenate
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
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # Heavy Augmentation for Training
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

    train_ds = PrescriptionDataset(os.path.join(args.data_dir, 'train'), transform=train_tfm)
    val_ds = PrescriptionDataset(os.path.join(args.data_dir, 'valid'), transform=val_tfm)
    
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              collate_fn=collate_mil_pad, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = GatedAttentionMIL(num_classes=len(train_ds.classes)).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        run_bag_corr = 0
        run_bag_total = 0
        
        for i, (images, labels, mask, _) in enumerate(train_loader):
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            bag_logits, inst_logits, _ = model(images, mask)
            
            # Loss Calculation
            loss_bag = criterion(bag_logits, labels)
            
            B, M, _ = inst_logits.shape
            expanded_labels = labels.unsqueeze(1).repeat(1, M)
            flat_inst_logits = inst_logits.view(B*M, -1)
            flat_labels = expanded_labels.view(-1)
            flat_mask = mask.view(-1)
            
            loss_inst = criterion(flat_inst_logits[flat_mask==1], flat_labels[flat_mask==1])
            total_loss = loss_bag + (0.5 * loss_inst)
            
            total_loss.backward()
            optimizer.step()
            
            # Metrics
            bag_preds = torch.argmax(bag_logits, dim=1)
            run_bag_corr += (bag_preds == labels).sum().item()
            run_bag_total += labels.size(0)

            if local_rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}] Loss: {total_loss.item():.4f} | Bag Acc: {run_bag_corr/run_bag_total:.2%}")

        scheduler.step()
        
        if local_rank == 0:
            print(f"--> Saving Checkpoint and Visuals for Epoch {epoch}")
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            save_visual_check(model, val_ds, device, epoch, ".")

    cleanup_ddp()

if __name__ == "__main__":
    main()
