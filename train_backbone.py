#!/usr/bin/env python3
"""
Phase 1: Backbone Training (Standard Classification)
=====================================================

Simple, fast training of ResNet34 for pill classification.
This is just standard image classification - nothing fancy.

Usage:
    # Single GPU
    python train_phase1_backbone.py --data-dir /path/to/data --epochs 30
    
    # Multi-GPU (recommended)
    torchrun --nproc_per_node=8 train_phase1_backbone.py --data-dir /path/to/data --epochs 30

Output:
    - backbone_best.pt: Best model weights
    - backbone_final.pt: Final model weights
    
Then run Phase 2:
    python train_phase2_verifier.py --data-dir /path/to/data --backbone backbone_best.pt
"""

import os
import sys
import re
import csv
import argparse
import random
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# Simple Flat Dataset (Standard ImageFolder-style)
# =============================================================================

class PillDataset(Dataset):
    """
    Simple flat dataset - each image is independent.
    No prescription grouping, no complexity.
    """
    
    def __init__(self, data_dir: str, index_file: str, split: str, 
                 class_to_idx: dict, input_size: int = 224, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.input_size = input_size
        
        # Load index
        self.samples = []
        with open(index_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.samples.append({
                        'path': row['relative_path'],
                        'label': class_to_idx[row['ndc']]
                    })
        
        # Transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            img_path = self.data_dir / sample['path']
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            # Fallback to black image (should rarely happen)
            img = torch.zeros(3, self.input_size, self.input_size)
        
        return img, sample['label']


def load_index(index_file: str):
    """Load index and build class mapping."""
    all_ndcs = set()
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ndcs.add(row['ndc'])
    
    class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
    idx_to_class = {idx: ndc for ndc, idx in class_to_idx.items()}
    
    return class_to_idx, idx_to_class


# =============================================================================
# Model
# =============================================================================

class PillClassifier(nn.Module):
    """Simple ResNet34 classifier."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        
        # Replace final FC
        self.backbone.fc = nn.Linear(512, num_classes)
        self.embed_dim = 512
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_embeddings(self, x):
        """Extract embeddings before final FC."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, device, epoch, rank):
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
    else:
        pbar = loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
    
    return {
        'loss': total_loss / total_samples,
        'acc': total_correct / total_samples
    }


@torch.no_grad()
def validate(model, loader, device, rank):
    model.eval()
    
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            logits = model(images)
        
        # Top-1
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        
        # Top-5
        _, top5_pred = logits.topk(5, dim=1)
        total_correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        
        total_samples += images.size(0)
    
    # Sync across GPUs
    if dist.is_initialized():
        stats = torch.tensor([total_correct, total_correct_top5, total_samples], 
                           device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        total_correct, total_correct_top5, total_samples = stats.tolist()
    
    return {
        'acc': total_correct / total_samples,
        'acc_top5': total_correct_top5 / total_samples
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Backbone Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output_phase1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # DDP setup
    is_distributed = 'RANK' in os.environ
    
    if is_distributed:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        print("=" * 60)
        print("PHASE 1: BACKBONE TRAINING")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size} × {world_size} = {args.batch_size * world_size}")
        print(f"Input size: {args.input_size}")
    
    # Load data
    index_file = os.path.join(args.data_dir, 'dataset_index.csv')
    class_to_idx, idx_to_class = load_index(index_file)
    num_classes = len(class_to_idx)
    
    if rank == 0:
        print(f"Classes: {num_classes}")
    
    train_dataset = PillDataset(
        args.data_dir, index_file, 'train', class_to_idx, 
        args.input_size, is_training=True
    )
    val_dataset = PillDataset(
        args.data_dir, index_file, 'valid', class_to_idx,
        args.input_size, is_training=False
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model = PillClassifier(num_classes).to(device)
    
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            if rank == 0:
                print(f"\nResuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model weights
            if is_distributed:
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
            
            # Load optimizer if available
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc = checkpoint.get('val_acc', 0.0)
            
            # Advance scheduler to correct position
            for _ in range(start_epoch):
                scheduler.step()
            
            if rank == 0:
                print(f"  Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")
        else:
            if rank == 0:
                print(f"Warning: Resume file not found: {args.resume}")
    
    # Training loop
    
    if rank == 0:
        print("\nStarting training...")
    
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch, rank)
        val_metrics = validate(model, val_loader, device, rank)
        scheduler.step()
        
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
            print(f"  Val   - Acc: {val_metrics['acc']:.4f}, Top5: {val_metrics['acc_top5']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                save_dict = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': val_metrics['acc'],
                    'val_acc_top5': val_metrics['acc_top5'],
                    'num_classes': num_classes,
                    'class_to_idx': class_to_idx
                }
                torch.save(save_dict, os.path.join(args.output_dir, 'backbone_best.pt'))
                print(f"  ✓ New best! Saved backbone_best.pt")
            
            # Save periodic
            if (epoch + 1) % 10 == 0:
                torch.save(save_dict, os.path.join(args.output_dir, f'backbone_epoch_{epoch}.pt'))
    
    # Save final
    if rank == 0:
        save_dict = {
            'epoch': args.epochs - 1,
            'model': model.module.state_dict() if is_distributed else model.state_dict(),
            'val_acc': val_metrics['acc'],
            'num_classes': num_classes,
            'class_to_idx': class_to_idx
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'backbone_final.pt'))
        
        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETE")
        print("=" * 60)
        print(f"Best validation accuracy: {best_acc:.4f}")
        print(f"Model saved to: {args.output_dir}/backbone_best.pt")
        print(f"\nNext step - Run Phase 2:")
        print(f"  python train_phase2_verifier.py --data-dir {args.data_dir} \\")
        print(f"      --backbone {args.output_dir}/backbone_best.pt")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()