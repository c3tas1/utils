#!/usr/bin/env python3
"""
FAST Verifier Training - DEBUG VERSION
=======================================

Added verbose logging at every step to find exactly where it hangs.
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
os.environ['NCCL_DEBUG'] = 'WARN'

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


def log(msg, rank=0):
    """Always print with timestamp and rank."""
    print(f"[{time.strftime('%H:%M:%S')}][R{rank}] {msg}", flush=True)


def setup():
    if 'RANK' not in os.environ:
        log("Single GPU mode", 0)
        return False, 0, 0, 1, torch.device('cuda:0')
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    log("Calling init_process_group...", rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    log("init_process_group DONE", rank)
    
    # Test
    log("Testing all_reduce...", rank)
    t = torch.ones(1, device=device)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    log(f"all_reduce OK, sum={t.item()}", rank)
    
    return True, rank, local_rank, world_size, device


# =============================================================================
# Embedding Store
# =============================================================================

class FastEmbeddingStore:
    def __init__(self, emb_dir, rank=0):
        self.dir = Path(emb_dir)
        self.rank = rank
        
        log("Loading index.json...", rank)
        with open(self.dir / 'index.json') as f:
            self.index = json.load(f)
        
        self.num_classes = self.index['num_classes']
        self.class_to_idx = self.index['class_to_idx']
        log(f"num_classes: {self.num_classes}", rank)
        
        all_emb = []
        all_pred = []
        all_conf = []
        path_to_idx = {}
        
        chunk_files = self.index['chunk_files']
        log(f"Loading {len(chunk_files)} chunks...", rank)
        
        for i, chunk_file in enumerate(chunk_files):
            if i % 10 == 0:
                log(f"  Chunk {i}/{len(chunk_files)}...", rank)
            
            data = np.load(self.dir / chunk_file, allow_pickle=True)
            embs = data['embeddings'].astype(np.float32)
            logits = data['logits'].astype(np.float32)
            paths = data['paths']
            
            for j, path in enumerate(paths):
                idx = len(all_emb)
                path_to_idx[str(path)] = idx
                all_emb.append(embs[j])
                
                pred = logits[j].argmax()
                conf = float(np.exp(logits[j][pred]) / (np.exp(logits[j]).sum() + 1e-8))
                all_pred.append(pred)
                all_conf.append(conf)
            
            data.close()
        
        log(f"Converting to tensors ({len(all_emb)} embeddings)...", rank)
        self.embeddings = torch.from_numpy(np.stack(all_emb))
        self.predictions = torch.tensor(all_pred, dtype=torch.long)
        self.confidences = torch.tensor(all_conf, dtype=torch.float32)
        self.path_to_idx = path_to_idx
        
        log(f"Embeddings shape: {self.embeddings.shape}", rank)
        log(f"Memory: {self.embeddings.nbytes / 1e9:.2f} GB", rank)


# =============================================================================
# Model
# =============================================================================

class FastVerifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        self.class_embed = nn.Embedding(num_classes, 64)
        
        self.pill_encoder = nn.Sequential(
            nn.Linear(EMBED_DIM + 64 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.rx_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.script_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.anomaly_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, embeddings, pred_classes, confidences, expected, mask):
        B, N, _ = embeddings.shape
        
        class_emb = self.class_embed(pred_classes)
        
        pill_feat = torch.cat([
            embeddings,
            class_emb,
            confidences.unsqueeze(-1)
        ], dim=-1)
        
        pill_tokens = self.pill_encoder(pill_feat)
        
        rx_feat = self.rx_encoder(expected)
        rx_query = rx_feat.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(pill_tokens, rx_query, rx_query)
        pill_tokens = pill_tokens + attn_out
        
        encoded = self.encoder(pill_tokens, src_key_padding_mask=mask)
        
        mask_expanded = (~mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        combined = torch.cat([pooled, rx_feat], dim=-1)
        script_logit = self.script_head(combined)
        anomaly_logits = self.anomaly_head(encoded).squeeze(-1)
        
        return script_logit, anomaly_logits


# =============================================================================
# Dataset
# =============================================================================

class FastRxDataset(Dataset):
    def __init__(self, store, prescriptions, error_rate=0.5, training=True, max_pills=100):
        self.store = store
        self.prescriptions = prescriptions
        self.error_rate = error_rate
        self.training = training
        self.max_pills = max_pills
        self.num_classes = store.num_classes
        self.class_to_idx = store.class_to_idx
        
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
        pills = rx['pills'][:self.max_pills]
        
        indices = [self.store.path_to_idx.get(p['path'], -1) for p in pills]
        
        expected = torch.zeros(self.num_classes)
        for p in pills:
            ndc_idx = self.class_to_idx.get(p['ndc'], 0)
            expected[ndc_idx] += 1
        expected = expected / max(len(pills), 1)
        
        is_correct = 1.0
        pill_correct = [1.0] * len(pills)
        
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

def sync_grads(model, world_size, rank):
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
    
    log("=" * 60, rank)
    log("FAST VERIFIER - DEBUG VERSION", rank)
    log("=" * 60, rank)
    
    emb_dir = Path(args.output_dir) / 'embeddings'
    
    # Load backbone info
    log("Loading backbone...", rank)
    ckpt = torch.load(args.backbone, map_location='cpu')
    num_classes = ckpt['num_classes']
    log(f"num_classes: {num_classes}", rank)
    
    # Load embeddings
    log("Loading embeddings...", rank)
    store = FastEmbeddingStore(emb_dir, rank)
    
    log("Moving embeddings to GPU...", rank)
    store.embeddings = store.embeddings.to(device)
    store.predictions = store.predictions.to(device)
    store.confidences = store.confidences.to(device)
    torch.cuda.synchronize()
    log("Embeddings on GPU", rank)
    
    # Barrier
    if is_dist:
        log("Barrier after embedding load...", rank)
        dist.barrier()
        log("Barrier passed", rank)
    
    # Load prescriptions
    log("Loading prescriptions...", rank)
    data_index = os.path.join(args.data_dir, 'dataset_index.csv')
    train_rx = load_prescriptions(data_index, 'train', max_pills=args.max_pills)
    val_rx = load_prescriptions(data_index, 'valid', max_pills=args.max_pills)
    log(f"Train: {len(train_rx)}, Val: {len(val_rx)}", rank)
    
    # Datasets
    log("Creating datasets...", rank)
    train_ds = FastRxDataset(store, train_rx, args.error_rate, True, args.max_pills)
    val_ds = FastRxDataset(store, val_rx, 0, False, args.max_pills)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist else None
    
    log("Creating dataloaders...", rank)
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
    log(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}", rank)
    
    # Model
    log("Creating model...", rank)
    model = FastVerifier(num_classes, args.hidden_dim).to(device)
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}", rank)
    
    # Sync weights
    if is_dist:
        log("Broadcasting weights...", rank)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        torch.cuda.synchronize()
        log("Weights synced", rank)
    
    # Optimizer
    log("Creating optimizer...", rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Training
    log("=" * 40, rank)
    log("STARTING TRAINING LOOP", rank)
    log("=" * 40, rank)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        log(f"Epoch {epoch} starting...", rank)
        
        if is_dist:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Log first few batches
            if batch_idx < 3 or batch_idx % 100 == 0:
                log(f"  Batch {batch_idx}/{len(train_loader)} - getting data...", rank)
            
            indices = batch['indices'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            expected = batch['expected'].to(device, non_blocking=True)
            is_correct = batch['is_correct'].to(device, non_blocking=True)
            pill_correct = batch['pill_correct'].to(device, non_blocking=True)
            
            if batch_idx < 3:
                log(f"  Batch {batch_idx} - looking up embeddings...", rank)
            
            B, N = indices.shape
            flat_idx = indices.clamp(min=0).view(-1)
            
            embeddings = store.embeddings[flat_idx].view(B, N, -1)
            pred_classes = store.predictions[flat_idx].view(B, N)
            confidences = store.confidences[flat_idx].view(B, N)
            
            invalid = (indices < 0)
            embeddings = embeddings.masked_fill(invalid.unsqueeze(-1), 0)
            pred_classes = pred_classes.masked_fill(invalid, 0)
            confidences = confidences.masked_fill(invalid, 0)
            
            if batch_idx < 3:
                log(f"  Batch {batch_idx} - forward pass...", rank)
            
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
                    anom_loss = torch.tensor(0.0, device=device)
                
                loss = script_loss + anom_loss
            
            if batch_idx < 3:
                log(f"  Batch {batch_idx} - backward...", rank)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            if batch_idx < 3:
                log(f"  Batch {batch_idx} - sync grads...", rank)
            
            sync_grads(model, world_size, rank)
            
            if batch_idx < 3:
                log(f"  Batch {batch_idx} - optimizer step...", rank)
            
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
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                avg_loss = total_loss / total_count
                avg_acc = total_correct / total_count
                log(f"  Batch {batch_idx}/{len(train_loader)}: loss={avg_loss:.4f}, acc={avg_acc:.4f}", rank)
        
        # End of epoch stats
        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        epoch_time = time.time() - epoch_start
        
        log(f"Epoch {epoch} train done: loss={train_loss:.4f}, acc={train_acc:.4f}, time={epoch_time:.1f}s", rank)
        
        # Validation
        log(f"Epoch {epoch} validation starting...", rank)
        model.eval()
        val_correct = 0
        val_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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
        
        log(f"Epoch {epoch} validation local: {val_correct}/{val_count}", rank)
        
        # Sync validation stats
        if is_dist:
            log(f"Epoch {epoch} syncing val stats...", rank)
            stats = torch.tensor([float(val_correct), float(val_count)], device=device)
            dist.all_reduce(stats)
            val_correct, val_count = int(stats[0].item()), int(stats[1].item())
            log(f"Epoch {epoch} synced: {val_correct}/{val_count}", rank)
        
        val_acc = val_correct / max(val_count, 1)
        
        scheduler.step()
        
        # Print results
        log("=" * 50, rank)
        log(f"EPOCH {epoch} RESULTS:", rank)
        log(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}", rank)
        log(f"  Val:   acc={val_acc:.4f}", rank)
        log(f"  Time:  {epoch_time:.1f}s", rank)
        log("=" * 50, rank)
        
        # Save
        if rank == 0:
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                log("  ✓ New best!", rank)
            
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
        
        # Epoch barrier
        if is_dist:
            log(f"Epoch {epoch} end barrier...", rank)
            dist.barrier()
            log(f"Epoch {epoch} barrier passed", rank)
    
    # Final
    if rank == 0:
        log("Saving final model...", rank)
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
    
    log("✓ TRAINING COMPLETE!", rank)


if __name__ == '__main__':
    main()