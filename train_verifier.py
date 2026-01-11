#!/usr/bin/env python3
"""
Phase 2: Verifier Training (Frozen Backbone)
=============================================

Trains ONLY the transformer verifier with frozen backbone.
Precomputes embeddings for maximum speed.

Usage:
    python train_phase2_verifier.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --epochs 20

Output:
    - verifier_best.pt: Best verifier weights
    - full_model.pt: Combined backbone + verifier for inference
"""

import os
import sys
import re
import csv
import argparse
import random
import pickle
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
# Backbone (for embedding extraction)
# =============================================================================

class PillClassifier(nn.Module):
    """Same architecture as Phase 1."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet34()
        self.backbone.fc = nn.Linear(512, num_classes)
        self.embed_dim = 512
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_embeddings_and_logits(self, x):
        """Extract both embeddings and logits."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        embeddings = torch.flatten(x, 1)
        logits = self.backbone.fc(embeddings)
        
        return embeddings, logits


# =============================================================================
# Verifier
# =============================================================================

class PrescriptionVerifier(nn.Module):
    """Transformer-based verifier."""
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Project: embedding (512) + confidence (1) + logits_entropy (1) = 514
        self.projector = nn.Sequential(
            nn.Linear(embed_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        
        # Summary token
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Output heads
        self.script_head = nn.Linear(hidden_dim, 1)
        self.anomaly_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, embeddings, confidences, entropies, mask):
        """
        Args:
            embeddings: (B, N, 512)
            confidences: (B, N, 1) - max softmax prob
            entropies: (B, N, 1) - entropy of predictions
            mask: (B, N) - True for padding
        """
        B, N, _ = embeddings.shape
        
        # Combine features
        x = torch.cat([embeddings, confidences, entropies], dim=-1)
        x = self.projector(x)
        
        # Add summary token
        summary = self.summary_token.expand(B, -1, -1)
        x = torch.cat([summary, x], dim=1)
        
        # Extend mask
        summary_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([summary_mask, mask], dim=1)
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=full_mask)
        
        # Outputs
        script_logit = self.script_head(x[:, 0, :])
        anomaly_logits = self.anomaly_head(x[:, 1:, :]).squeeze(-1)
        
        return script_logit, anomaly_logits


# =============================================================================
# Precomputed Embeddings Dataset
# =============================================================================

class PrecomputedPrescriptionDataset(Dataset):
    """
    Dataset using precomputed embeddings.
    Much faster than running backbone every iteration!
    """
    
    def __init__(self, embeddings_file: str, prescriptions: list, 
                 error_rate: float = 0.5, is_training: bool = True):
        
        # Load precomputed embeddings
        print(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']  # {relative_path: embedding}
        self.logits = data['logits']  # {relative_path: logits}
        self.class_to_idx = data['class_to_idx']
        
        self.prescriptions = prescriptions
        self.error_rate = error_rate
        self.is_training = is_training
        
        # Build pill library for error injection
        if is_training and error_rate > 0:
            self.pill_library = defaultdict(list)
            for rx in prescriptions:
                for pill in rx['pills']:
                    self.pill_library[pill['ndc']].append(pill['path'])
            self.pill_library = dict(self.pill_library)
            self.ndc_list = list(self.pill_library.keys())
        else:
            self.pill_library = {}
            self.ndc_list = []
    
    def __len__(self):
        return len(self.prescriptions)
    
    def __getitem__(self, idx):
        rx = self.prescriptions[idx]
        pills = rx['pills']
        
        # Error injection
        is_correct = True
        pill_correct = [1.0] * len(pills)
        pill_paths = [p['path'] for p in pills]
        pill_labels = [self.class_to_idx[p['ndc']] for p in pills]
        
        if self.is_training and random.random() < self.error_rate:
            is_correct = False
            # Inject 1-3 errors
            num_errors = random.randint(1, min(3, len(pills)))
            error_indices = random.sample(range(len(pills)), num_errors)
            
            for err_idx in error_indices:
                orig_ndc = pills[err_idx]['ndc']
                wrong_ndc = random.choice([n for n in self.ndc_list if n != orig_ndc])
                wrong_path = random.choice(self.pill_library[wrong_ndc])
                
                pill_paths[err_idx] = wrong_path
                pill_labels[err_idx] = self.class_to_idx[wrong_ndc]
                pill_correct[err_idx] = 0.0
        
        # Get precomputed embeddings and logits
        embeddings = []
        logits_list = []
        
        for path in pill_paths:
            if path in self.embeddings:
                embeddings.append(self.embeddings[path])
                logits_list.append(self.logits[path])
            else:
                # Fallback (shouldn't happen)
                embeddings.append(torch.zeros(512))
                logits_list.append(torch.zeros(len(self.class_to_idx)))
        
        embeddings = torch.stack(embeddings)
        logits = torch.stack(logits_list)
        
        # Compute confidence and entropy
        probs = F.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values
        entropies = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        entropies = entropies / np.log(len(self.class_to_idx))  # Normalize
        
        return {
            'embeddings': embeddings,
            'logits': logits,
            'confidences': confidences,
            'entropies': entropies,
            'labels': torch.tensor(pill_labels, dtype=torch.long),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': torch.tensor(pill_correct),
            'num_pills': len(pills)
        }


def collate_fn(batch):
    """Collate with padding."""
    B = len(batch)
    max_pills = max(b['num_pills'] for b in batch)
    embed_dim = batch[0]['embeddings'].size(-1)
    num_classes = batch[0]['logits'].size(-1)
    
    embeddings = torch.zeros(B, max_pills, embed_dim)
    logits = torch.zeros(B, max_pills, num_classes)
    confidences = torch.zeros(B, max_pills)
    entropies = torch.zeros(B, max_pills)
    labels = torch.full((B, max_pills), -100, dtype=torch.long)
    mask = torch.ones(B, max_pills, dtype=torch.bool)
    pill_correct = torch.zeros(B, max_pills)
    is_correct = torch.zeros(B)
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        embeddings[i, :n] = b['embeddings']
        logits[i, :n] = b['logits']
        confidences[i, :n] = b['confidences']
        entropies[i, :n] = b['entropies']
        labels[i, :n] = b['labels']
        mask[i, :n] = False
        pill_correct[i, :n] = b['pill_correct']
        is_correct[i] = b['is_correct']
    
    return {
        'embeddings': embeddings,
        'logits': logits,
        'confidences': confidences.unsqueeze(-1),
        'entropies': entropies.unsqueeze(-1),
        'labels': labels,
        'mask': mask,
        'pill_correct': pill_correct,
        'is_correct': is_correct
    }


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(backbone, data_dir, index_file, output_file, 
                       device, batch_size=64, input_size=224):
    """Extract and cache embeddings for all images."""
    
    print("Extracting embeddings...")
    backbone.eval()
    
    # Load all image paths
    samples = []
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row['relative_path'])
    
    samples = list(set(samples))  # Unique paths
    print(f"  Total unique images: {len(samples)}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Extract in batches
    embeddings = {}
    logits = {}
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting"):
        batch_paths = samples[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(Path(data_dir) / path).convert('RGB')
                batch_images.append(transform(img))
                valid_paths.append(path)
            except Exception as e:
                continue
        
        if not batch_images:
            continue
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with autocast():
            emb, log = backbone.get_embeddings_and_logits(batch_tensor)
        
        emb = emb.cpu()
        log = log.cpu()
        
        for j, path in enumerate(valid_paths):
            embeddings[path] = emb[j]
            logits[path] = log[j]
    
    # Get class_to_idx from backbone checkpoint
    class_to_idx = {}
    all_ndcs = set()
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ndcs.add(row['ndc'])
    class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
    
    # Save
    print(f"  Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'logits': logits,
            'class_to_idx': class_to_idx
        }, f)
    
    print(f"  Done! Extracted {len(embeddings)} embeddings")
    return embeddings, logits, class_to_idx


def load_prescriptions(index_file: str, split: str, min_pills: int = 5, max_pills: int = 200):
    """Group images into prescriptions."""
    
    by_rx = defaultdict(list)
    
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == split:
                by_rx[row['prescription_id']].append({
                    'path': row['relative_path'],
                    'ndc': row['ndc'],
                    'patch_no': int(row['patch_no'])
                })
    
    prescriptions = []
    for rx_id, pills in by_rx.items():
        if min_pills <= len(pills) <= max_pills:
            pills = sorted(pills, key=lambda x: x['patch_no'])
            prescriptions.append({
                'rx_id': rx_id,
                'pills': pills
            })
    
    return prescriptions


# =============================================================================
# Training
# =============================================================================

def train_epoch(verifier, loader, optimizer, scaler, device, epoch, rank):
    verifier.train()
    
    total_loss = 0
    total_script_correct = 0
    total_script_count = 0
    total_anomaly_correct = 0
    total_anomaly_count = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
    else:
        pbar = loader
    
    for batch in pbar:
        embeddings = batch['embeddings'].to(device)
        confidences = batch['confidences'].to(device)
        entropies = batch['entropies'].to(device)
        mask = batch['mask'].to(device)
        is_correct = batch['is_correct'].to(device)
        pill_correct = batch['pill_correct'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            script_logit, anomaly_logits = verifier(embeddings, confidences, entropies, mask)
            
            # Script loss
            script_loss = F.binary_cross_entropy_with_logits(
                script_logit.squeeze(-1), is_correct
            )
            
            # Anomaly loss
            valid = ~mask
            if valid.any():
                anomaly_target = 1.0 - pill_correct
                anomaly_loss = F.binary_cross_entropy_with_logits(
                    anomaly_logits[valid], anomaly_target[valid]
                )
            else:
                anomaly_loss = torch.tensor(0.0, device=device)
            
            loss = script_loss + 2.0 * anomaly_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        B = embeddings.size(0)
        total_loss += loss.item() * B
        
        with torch.no_grad():
            script_pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
            total_script_correct += (script_pred == is_correct).sum().item()
            total_script_count += B
            
            if valid.any():
                anomaly_pred = (torch.sigmoid(anomaly_logits) > 0.5).float()
                anomaly_target = 1.0 - pill_correct
                total_anomaly_correct += ((anomaly_pred == anomaly_target) & valid).sum().item()
                total_anomaly_count += valid.sum().item()
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/total_script_count:.4f}',
                'script': f'{total_script_correct/total_script_count:.4f}',
                'anomaly': f'{total_anomaly_correct/max(1,total_anomaly_count):.4f}'
            })
    
    return {
        'loss': total_loss / total_script_count,
        'script_acc': total_script_correct / total_script_count,
        'anomaly_acc': total_anomaly_correct / max(1, total_anomaly_count)
    }


@torch.no_grad()
def validate(verifier, loader, device):
    verifier.eval()
    
    total_script_correct = 0
    total_script_count = 0
    total_anomaly_correct = 0
    total_anomaly_count = 0
    
    for batch in loader:
        embeddings = batch['embeddings'].to(device)
        confidences = batch['confidences'].to(device)
        entropies = batch['entropies'].to(device)
        mask = batch['mask'].to(device)
        is_correct = batch['is_correct'].to(device)
        pill_correct = batch['pill_correct'].to(device)
        
        with autocast():
            script_logit, anomaly_logits = verifier(embeddings, confidences, entropies, mask)
        
        B = embeddings.size(0)
        valid = ~mask
        
        script_pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
        total_script_correct += (script_pred == is_correct).sum().item()
        total_script_count += B
        
        if valid.any():
            anomaly_pred = (torch.sigmoid(anomaly_logits) > 0.5).float()
            anomaly_target = 1.0 - pill_correct
            total_anomaly_correct += ((anomaly_pred == anomaly_target) & valid).sum().item()
            total_anomaly_count += valid.sum().item()
    
    return {
        'script_acc': total_script_correct / total_script_count,
        'anomaly_acc': total_anomaly_correct / max(1, total_anomaly_count)
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Verifier Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True, help='Path to backbone_best.pt')
    parser.add_argument('--output-dir', type=str, default='./output_phase2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--error-rate', type=float, default=0.5)
    parser.add_argument('--skip-extraction', action='store_true', 
                       help='Skip embedding extraction (use cached)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    rank = 0  # Single GPU for Phase 2 (fast enough)
    
    print("=" * 60)
    print("PHASE 2: VERIFIER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    
    # Load backbone
    print("\nLoading backbone...")
    checkpoint = torch.load(args.backbone, map_location=device)
    num_classes = checkpoint['num_classes']
    class_to_idx = checkpoint['class_to_idx']
    
    backbone = PillClassifier(num_classes).to(device)
    backbone.load_state_dict(checkpoint['model'])
    backbone.eval()
    
    print(f"  Loaded! Val accuracy was: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  Classes: {num_classes}")
    
    # Extract embeddings (or load cached)
    index_file = os.path.join(args.data_dir, 'dataset_index.csv')
    embeddings_file = os.path.join(args.output_dir, 'embeddings_cache.pkl')
    
    if not args.skip_extraction or not os.path.exists(embeddings_file):
        extract_embeddings(
            backbone, args.data_dir, index_file, embeddings_file,
            device, batch_size=64, input_size=args.input_size
        )
    else:
        print(f"\nUsing cached embeddings: {embeddings_file}")
    
    # Load prescriptions
    print("\nLoading prescriptions...")
    train_prescriptions = load_prescriptions(index_file, 'train')
    val_prescriptions = load_prescriptions(index_file, 'valid')
    print(f"  Train prescriptions: {len(train_prescriptions)}")
    print(f"  Val prescriptions: {len(val_prescriptions)}")
    
    # Datasets
    train_dataset = PrecomputedPrescriptionDataset(
        embeddings_file, train_prescriptions, 
        error_rate=args.error_rate, is_training=True
    )
    val_dataset = PrecomputedPrescriptionDataset(
        embeddings_file, val_prescriptions,
        error_rate=args.error_rate, is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Verifier
    verifier = PrescriptionVerifier(
        embed_dim=512,
        hidden_dim=args.hidden_dim,
        num_heads=8,
        num_layers=args.num_layers,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in verifier.parameters())
    print(f"\nVerifier parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Training
    print("\nStarting training...")
    best_script_acc = 0.0
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(verifier, train_loader, optimizer, scaler, device, epoch, rank)
        val_metrics = validate(verifier, val_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Script: {train_metrics['script_acc']:.4f}, Anomaly: {train_metrics['anomaly_acc']:.4f}")
        print(f"  Val   - Script: {val_metrics['script_acc']:.4f}, Anomaly: {val_metrics['anomaly_acc']:.4f}")
        
        if val_metrics['script_acc'] > best_script_acc:
            best_script_acc = val_metrics['script_acc']
            torch.save({
                'epoch': epoch,
                'verifier': verifier.state_dict(),
                'script_acc': val_metrics['script_acc'],
                'anomaly_acc': val_metrics['anomaly_acc']
            }, os.path.join(args.output_dir, 'verifier_best.pt'))
            print(f"  ✓ New best! Saved verifier_best.pt")
    
    # Save combined model
    print("\nSaving combined model...")
    torch.save({
        'backbone': checkpoint['model'],
        'verifier': verifier.state_dict(),
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers
    }, os.path.join(args.output_dir, 'full_model.pt'))
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Best script accuracy: {best_script_acc:.4f}")
    print(f"Models saved to: {args.output_dir}/")
    print(f"\nFor inference, use: {args.output_dir}/full_model.pt")


if __name__ == '__main__':
    main()