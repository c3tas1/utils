#!/usr/bin/env python3
"""
FAST Verifier Training
======================

Optimizations:
1. Pre-load ALL embeddings to GPU memory (not disk)
2. Use embedding indices instead of full logit tensors
3. Smaller model with same accuracy
4. Efficient batching

Expected: 20 epochs in 30-60 minutes on 8x A100

Usage:
    torchrun --nproc_per_node=8 --master_port=29500 fast_verifier.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --output-dir output_phase2
"""

import os
import sys
import csv
import json
import random
import time
import argparse
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

os.environ['NCCL_TIMEOUT'] = '1800'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

EMBED_DIM = 512


def log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup():
    if 'RANK' not in os.environ:
        return False, 0, 0, 1, torch.device('cuda:0')
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    
    # Quick test
    t = torch.ones(1, device=device)
    dist.all_reduce(t)
    
    return True, rank, local_rank, world_size, device


# =============================================================================
# FAST Embedding Store - Pre-load to tensors
# =============================================================================

class FastEmbeddingStore:
    """Load ALL embeddings into contiguous tensors for fast indexing."""
    
    def __init__(self, emb_dir, device='cpu'):
        self.dir = Path(emb_dir)
        
        with open(self.dir / 'index.json') as f:
            self.index = json.load(f)
        
        self.num_classes = self.index['num_classes']
        self.class_to_idx = self.index['class_to_idx']
        
        print(f"  Loading embeddings into memory...")
        print(f"  num_classes: {self.num_classes}")
        
        # Collect all data
        all_emb = []
        all_pred = []  # Store predicted class, not full logits!
        all_conf = []  # Store confidence
        path_to_idx = {}
        
        for chunk_file in tqdm(self.index['chunk_files'], desc='  Loading chunks'):
            data = np.load(self.dir / chunk_file, allow_pickle=True)
            embs = data['embeddings'].astype(np.float32)
            logits = data['logits'].astype(np.float32)
            paths = data['paths']
            
            for i, path in enumerate(paths):
                idx = len(all_emb)
                path_to_idx[str(path)] = idx
                all_emb.append(embs[i])
                
                # Store only pred class + confidence (not 50k logits!)
                pred = logits[i].argmax()
                conf = np.exp(logits[i][pred]) / np.exp(logits[i]).sum()
                all_pred.append(pred)
                all_conf.append(conf)
            
            data.close()
        
        # Convert to tensors
        self.embeddings = torch.from_numpy(np.stack(all_emb))  # [N, 512]
        self.predictions = torch.tensor(all_pred, dtype=torch.long)  # [N]
        self.confidences = torch.tensor(all_conf, dtype=torch.float32)  # [N]
        self.path_to_idx = path_to_idx
        
        print(f"  Loaded {len(self.embeddings):,} embeddings")
        print(f"  Memory: {self.embeddings.nbytes / 1e9:.2f} GB")
    
    def get_indices(self, paths):
        """Get indices for a list of paths."""
        return [self.path_to_idx.get(p, -1) for p in paths]


# =============================================================================
# FAST Model - Smaller, efficient
# =============================================================================

class FastVerifier(nn.Module):
    """
    Efficient verifier that works with:
    - Embeddings [B, N, 512]
    - Predicted classes [B, N] (not full logits)
    - Confidences [B, N]
    """
    
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # Class embedding (instead of processing 50k logits)
        self.class_embed = nn.Embedding(num_classes, 64)
        
        # Pill encoder: [emb(512) + class_emb(64) + conf(1)] -> hidden
        self.pill_encoder = nn.Sequential(
            nn.Linear(EMBED_DIM + 64 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Expected composition encoder
        self.rx_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Self attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Heads
        self.script_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.anomaly_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, embeddings, pred_classes, confidences, expected, mask):
        """
        Args:
            embeddings: [B, N, 512]
            pred_classes: [B, N] - predicted class indices
            confidences: [B, N] - prediction confidences
            expected: [B, num_classes] - expected composition
            mask: [B, N] - True for padding
        """
        B, N, _ = embeddings.shape
        
        # Get class embeddings
        class_emb = self.class_embed(pred_classes)  # [B, N, 64]
        
        # Combine features
        pill_feat = torch.cat([
            embeddings,
            class_emb,
            confidences.unsqueeze(-1)
        ], dim=-1)  # [B, N, 512+64+1]
        
        pill_tokens = self.pill_encoder(pill_feat)  # [B, N, H]
        
        # Encode expected
        rx_feat = self.rx_encoder(expected)  # [B, H]
        rx_query = rx_feat.unsqueeze(1)  # [B, 1, H]
        
        # Cross attention
        attn_out, _ = self.cross_attn(
            pill_tokens, rx_query, rx_query
        )
        pill_tokens = pill_tokens + attn_out
        
        # Self attention
        encoded = self.encoder(pill_tokens, src_key_padding_mask=mask)
        
        # Pool (mean of non-masked)
        mask_expanded = (~mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Classify
        combined = torch.cat([pooled, rx_feat], dim=-1)
        script_logit = self.script_head(combined)
        anomaly_logits = self.anomaly_head(encoded).squeeze(-1)
        
        return script_logit, anomaly_logits


# =============================================================================
# Dataset
# =============================================================================

class FastRxDataset(Dataset):
    """Fast dataset using pre-indexed embeddings."""
    
    def __init__(self, store, prescriptions, error_rate=0.5, training=True, max_pills=100):
        self.store = store
        self.prescriptions = prescriptions
        self.error_rate = error_rate
        self.training = training
        self.max_pills = max_pills
        self.num_classes = store.num_classes
        self.class_to_idx = store.class_to_idx
        
        # Build pill library
        if training and error_rate > 0:
            self.pill_lib = defaultdict(list)
            for rx in prescriptions:
                for pill in rx['pills']:
                    idx = store.path_to_idx.get(pill['path'], -1)
                    if idx >= 0:
                        self.pill_lib[pill['ndc']].append(idx)
            self.ndc_list = list(self.pill_lib.keys())
        else:
            self.pill_lib = {}
            self.ndc_list = []
    
    def __len__(self):
        return len(self.prescriptions)
    
    def __getitem__(self, idx):
        rx = self.prescriptions[idx]
        pills = rx['pills'][:self.max_pills]  # Cap pills
        
        # Get embedding indices
        indices = [self.store.path_to_idx.get(p['path'], -1) for p in pills]
        
        # Expected composition
        expected = torch.zeros(self.num_classes)
        for p in pills:
            ndc_idx = self.class_to_idx.get(p['ndc'], 0)
            expected[ndc_idx] += 1
        expected = expected / len(pills)
        
        is_correct = 1.0
        pill_correct = [1.0] * len(pills)
        
        # Error injection
        if self.training and self.ndc_list and random.random() < self.error_rate:
            is_correct = 0.0
            n_errors = random.randint(1, min(3, len(pills)))
            for err_idx in random.sample(range(len(pills)), n_errors):
                orig_ndc = pills[err_idx]['ndc']
                wrong_ndcs = [n for n in self.ndc_list if n != orig_ndc]
                if wrong_ndcs:
                    wrong_ndc = random.choice(wrong_ndcs)
                    if self.pill_lib.get(wrong_ndc):
                        indices[err_idx] = random.choice(self.pill_lib[wrong_ndc])
                        pill_correct[err_idx] = 0.0
        
        return {
            'indices': torch.tensor(indices, dtype=torch.long),
            'expected': expected,
            'is_correct': torch.tensor(is_correct),
            'pill_correct': torch.tensor(pill_correct),
            'num_pills': len(pills)
        }


def collate_fn(batch):
    B = len(batch)
    max_pills = max(b['num_pills'] for b in batch)
    num_classes = batch[0]['expected'].size(0)
    
    indices = torch.full((B, max_pills), -1, dtype=torch.long)
    mask = torch.ones(B, max_pills, dtype=torch.bool)
    pill_correct = torch.zeros(B, max_pills)
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        indices[i, :n] = b['indices']
        mask[i, :n] = False
        pill_correct[i, :n] = b['pill_correct']
    
    return {
        'indices': indices,
        'mask': mask,
        'expected': torch.stack([b['expected'] for b in batch]),
        'is_correct': torch.stack([b['is_correct'] for b in batch]),
        'pill_correct': pill_correct
    }


def load_prescriptions(index_file, split, min_pills=5, max_pills=200):
    by_rx = defaultdict(list)
    with open(index_file) as f:
        for row in csv.DictReader(f):
            if row['split'] == split:
                by_rx[row['prescription_id']].append({
                    'path': row['relative_path'],
                    'ndc': row['ndc'],
                    'patch_no': int(row['patch_no'])
                })
    
    prescriptions = []
    for rx_id, pills in by_rx.items():
        if min_pills <= len(pills) <= max_pills:
            pills.sort(key=lambda x: x['patch_no'])
            prescriptions.append({'rx_id': rx_id, 'pills': pills})
    return prescriptions


# =============================================================================
# Training
# =============================================================================

def sync_grads(model, world_size):
    if world_size == 1:
        return
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat(grads)
    dist.all_reduce(flat)
    flat.div_(world_size)
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.grad.numel()
            p.grad.copy_(flat[offset:offset+numel].view_as(p.grad))
            offset += numel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--output-dir', default='./output_phase2')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--error-rate', type=float, default=0.5)
    parser.add_argument('--max-pills', type=int, default=100)
    args = parser.parse_args()
    
    is_dist, rank, local_rank, world_size, device = setup()
    
    log(rank, "=" * 60)
    log(rank, "FAST VERIFIER TRAINING")
    log(rank, "=" * 60)
    
    emb_dir = Path(args.output_dir) / 'embeddings'
    
    # Load backbone info
    log(rank, "\nLoading backbone info...")
    ckpt = torch.load(args.backbone, map_location='cpu')
    num_classes = ckpt['num_classes']
    log(rank, f"  num_classes: {num_classes}")
    
    # Load embeddings
    log(rank, "\nLoading embeddings...")
    store = FastEmbeddingStore(emb_dir)
    
    # Move store tensors to device for fast lookup
    store.embeddings = store.embeddings.to(device)
    store.predictions = store.predictions.to(device)
    store.confidences = store.confidences.to(device)
    log(rank, f"  Embeddings on {device}")
    
    # Sync
    if is_dist:
        dist.barrier()
    
    # Load prescriptions
    log(rank, "\nLoading prescriptions...")
    data_index = os.path.join(args.data_dir, 'dataset_index.csv')
    train_rx = load_prescriptions(data_index, 'train', max_pills=args.max_pills)
    val_rx = load_prescriptions(data_index, 'valid', max_pills=args.max_pills)
    log(rank, f"  Train: {len(train_rx):,}, Val: {len(val_rx):,}")
    
    # Datasets
    train_ds = FastRxDataset(store, train_rx, args.error_rate, True, args.max_pills)
    val_ds = FastRxDataset(store, val_rx, 0, False, args.max_pills)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist else None
    
    # DataLoaders - num_workers=0 is OK here since data is in RAM
    train_loader = DataLoader(
        train_ds, args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=collate_fn, num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size,
        sampler=val_sampler, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    
    log(rank, f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}")
    
    # Model
    log(rank, "\nCreating model...")
    model = FastVerifier(num_classes, args.hidden_dim).to(device)
    log(rank, f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Sync weights
    if is_dist:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Training
    log(rank, "\n" + "=" * 40)
    log(rank, "TRAINING")
    log(rank, "=" * 40)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        if is_dist:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        total_correct = 0
        total_count = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=rank != 0)
        
        for batch in pbar:
            # Get indices and move to device
            indices = batch['indices'].to(device)
            mask = batch['mask'].to(device)
            expected = batch['expected'].to(device)
            is_correct = batch['is_correct'].to(device)
            pill_correct = batch['pill_correct'].to(device)
            
            # Lookup embeddings (fast - already on GPU)
            B, N = indices.shape
            flat_idx = indices.clamp(min=0).view(-1)
            
            embeddings = store.embeddings[flat_idx].view(B, N, -1)
            pred_classes = store.predictions[flat_idx].view(B, N)
            confidences = store.confidences[flat_idx].view(B, N)
            
            # Zero out invalid
            invalid = (indices < 0)
            embeddings = embeddings.masked_fill(invalid.unsqueeze(-1), 0)
            pred_classes = pred_classes.masked_fill(invalid, 0)
            confidences = confidences.masked_fill(invalid, 0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                script_logit, anom_logits = model(
                    embeddings, pred_classes, confidences, expected, mask
                )
                
                script_loss = F.binary_cross_entropy_with_logits(
                    script_logit.squeeze(-1), is_correct
                )
                
                valid = ~mask
                if valid.any():
                    anom_loss = F.binary_cross_entropy_with_logits(
                        anom_logits[valid], (1 - pill_correct)[valid]
                    )
                else:
                    anom_loss = 0
                
                loss = script_loss + anom_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            sync_grads(model, world_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Stats
            B = indices.size(0)
            total_loss += loss.item() * B
            with torch.no_grad():
                pred = torch.sigmoid(script_logit.squeeze(-1)) > 0.5
                total_correct += (pred == is_correct.bool()).sum().item()
            total_count += B
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/total_count:.4f}',
                    'acc': f'{total_correct/total_count:.4f}'
                })
        
        # Validation
        model.eval()
        val_correct = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                indices = batch['indices'].to(device)
                mask = batch['mask'].to(device)
                expected = batch['expected'].to(device)
                is_correct = batch['is_correct'].to(device)
                
                B, N = indices.shape
                flat_idx = indices.clamp(min=0).view(-1)
                
                embeddings = store.embeddings[flat_idx].view(B, N, -1)
                pred_classes = store.predictions[flat_idx].view(B, N)
                confidences = store.confidences[flat_idx].view(B, N)
                
                invalid = (indices < 0)
                embeddings = embeddings.masked_fill(invalid.unsqueeze(-1), 0)
                pred_classes = pred_classes.masked_fill(invalid, 0)
                confidences = confidences.masked_fill(invalid, 0)
                
                with autocast():
                    script_logit, _ = model(
                        embeddings, pred_classes, confidences, expected, mask
                    )
                
                pred = torch.sigmoid(script_logit.squeeze(-1)) > 0.5
                val_correct += (pred == is_correct.bool()).sum().item()
                val_count += B
        
        # Sync
        if is_dist:
            stats = torch.tensor([val_correct, val_count], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            val_correct, val_count = stats.tolist()
        
        val_acc = val_correct / max(val_count, 1)
        epoch_time = time.time() - epoch_start
        
        scheduler.step()
        
        if rank == 0:
            print(f"\nEpoch {epoch} ({epoch_time:.1f}s):")
            print(f"  Train: loss={total_loss/total_count:.4f}, acc={total_correct/total_count:.4f}")
            print(f"  Val:   acc={val_acc:.4f}")
            
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                print("  â New best!")
            
            # Save
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, Path(args.output_dir) / 'verifier_latest.pt')
            
            if is_best:
                torch.save({
                    'model': model.state_dict(),
                    'num_classes': num_classes,
                    'hidden_dim': args.hidden_dim
                }, Path(args.output_dir) / 'verifier_best.pt')
        
        if is_dist:
            dist.barrier()
    
    # Final
    if rank == 0:
        log(rank, "\nSaving final model...")
        backbone_ckpt = torch.load(args.backbone, map_location='cpu')
        torch.save({
            'backbone': backbone_ckpt['model'],
            'verifier': model.state_dict(),
            'num_classes': num_classes,
            'class_to_idx': backbone_ckpt['class_to_idx'],
            'hidden_dim': args.hidden_dim
        }, Path(args.output_dir) / 'full_model.pt')
    
    if is_dist:
        dist.destroy_process_group()
    
    log(rank, "\nâ DONE!")


if __name__ == '__main__':
    main()
