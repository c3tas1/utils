#!/usr/bin/env python3
"""
Phase 2: Consolidated Embedding Extraction + Verifier Training
==============================================================

Single file that handles:
1. Parallel embedding extraction (8 GPUs)
2. Verifier training with prescription-level info (8 GPUs)

Robust DDP with:
- Automatic NCCL timeout handling
- Fallback to gloo backend
- Checkpoint every N batches
- Auto-resume on restart
- Graceful interrupt handling

Usage:
    # Step 1: Extract embeddings
    torchrun --nproc_per_node=8 --master_port=29500 phase2_consolidated.py extract \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --output-dir output_phase2
    
    # Step 2: Train verifier
    torchrun --nproc_per_node=8 --master_port=29500 phase2_consolidated.py train \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --output-dir output_phase2
    
    # Or do both in sequence
    torchrun --nproc_per_node=8 --master_port=29500 phase2_consolidated.py all \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --output-dir output_phase2
"""

import os
import sys
import csv
import argparse
import json
import random
import signal
import time
import gc
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from functools import wraps

# Set NCCL env vars BEFORE importing torch
os.environ.setdefault('NCCL_TIMEOUT', '3600')
os.environ.setdefault('NCCL_IB_TIMEOUT', '60')
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '0')
os.environ.setdefault('NCCL_DEBUG', 'WARN')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Constants
# =============================================================================
EMBED_DIM = 512
CHUNK_SIZE = 100000  # Images per chunk


# =============================================================================
# Robust DDP Utilities
# =============================================================================

class GracefulKiller:
    """Handle interrupts gracefully."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, *args):
        self.kill_now = True


def setup_distributed():
    """Initialize distributed training with fallback."""
    if 'RANK' not in os.environ:
        return False, 0, 0, 1, torch.device('cuda:0')
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Try NCCL first, fallback to gloo
    for backend in ['nccl', 'gloo']:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=60),
                world_size=world_size,
                rank=rank
            )
            
            # Test communication
            test = torch.ones(1, device=device)
            dist.all_reduce(test)
            
            if rank == 0:
                print(f"  â Initialized {backend.upper()}")
            return True, rank, local_rank, world_size, device
            
        except Exception as e:
            if rank == 0:
                print(f"  â {backend} failed: {e}")
    
    # Fallback to single GPU
    if rank == 0:
        print("  â  Running single GPU mode")
    return False, 0, 0, 1, device


def safe_barrier():
    """Barrier with timeout handling."""
    if not dist.is_initialized():
        return
    try:
        dist.barrier()
    except Exception as e:
        print(f"  Warning: barrier failed: {e}")


def safe_all_reduce(tensor):
    """All-reduce with error handling."""
    if not dist.is_initialized():
        return
    try:
        dist.all_reduce(tensor)
    except Exception as e:
        print(f"  Warning: all_reduce failed: {e}")


def print_rank0(msg, rank=0):
    """Print only from rank 0."""
    if rank == 0:
        print(msg)


# =============================================================================
# Models
# =============================================================================

class PillClassifier(nn.Module):
    """ResNet34 backbone for pill classification."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet34()
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def get_embeddings_and_logits(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        emb = torch.flatten(x, 1)
        logits = self.backbone.fc(emb)
        return emb, logits


class PrescriptionVerifier(nn.Module):
    """Prescription-aware verifier with cross-attention."""
    
    def __init__(self, num_classes, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        self.ndc_embed = nn.Embedding(num_classes, hidden_dim)
        
        self.pill_proj = nn.Sequential(
            nn.Linear(EMBED_DIM + 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.rx_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout,
                batch_first=True, norm_first=True
            ),
            num_layers
        )
        
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.script_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.anomaly_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, embeddings, logits, expected_comp, mask):
        B, N, _ = embeddings.shape
        device = embeddings.device
        
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values.unsqueeze(-1)
        ent = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        ent = ent / np.log(self.num_classes)
        
        pred_ndc = logits.argmax(dim=-1)
        pred_emb = self.ndc_embed(pred_ndc)
        
        pill_feat = torch.cat([embeddings, conf, ent, pred_emb], dim=-1)
        pill_tokens = self.pill_proj(pill_feat)
        
        rx_enc = self.rx_encoder(expected_comp)
        rx_tokens = rx_enc.unsqueeze(1)
        
        attended, _ = self.cross_attn(pill_tokens, rx_tokens, rx_tokens)
        pill_tokens = pill_tokens + attended
        
        summary = self.summary_token.expand(B, -1, -1)
        tokens = torch.cat([summary, pill_tokens], dim=1)
        
        full_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            mask
        ], dim=1)
        
        encoded = self.encoder(tokens, src_key_padding_mask=full_mask)
        
        script_feat = torch.cat([encoded[:, 0], rx_enc], dim=-1)
        script_logit = self.script_head(script_feat)
        anomaly_logits = self.anomaly_head(encoded[:, 1:]).squeeze(-1)
        
        return script_logit, anomaly_logits


# =============================================================================
# Datasets
# =============================================================================

class ImageDataset(Dataset):
    """Dataset for embedding extraction."""
    
    def __init__(self, data_dir, paths, input_size=224):
        self.data_dir = Path(data_dir)
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(self.data_dir / path).convert('RGB')
            return self.transform(img), idx, True
        except:
            return torch.zeros(3, 224, 224), idx, False


class EmbeddingStore:
    """Memory-efficient embedding storage with LRU cache."""
    
    def __init__(self, embeddings_dir, cache_size=20):
        self.dir = Path(embeddings_dir)
        
        with open(self.dir / 'index.json') as f:
            self.index = json.load(f)
        
        self.num_classes = self.index['num_classes']
        self.class_to_idx = self.index['class_to_idx']
        
        # Build path -> location map
        self.path_to_loc = {}
        for chunk_file in self.index['chunk_files']:
            data = np.load(self.dir / chunk_file, allow_pickle=True)
            for i, p in enumerate(data['paths']):
                self.path_to_loc[str(p)] = (chunk_file, i)
            data.close()
        
        self._cache = {}
        self._cache_order = []
        self.cache_size = cache_size
    
    def _load_chunk(self, chunk_file):
        if chunk_file in self._cache:
            self._cache_order.remove(chunk_file)
            self._cache_order.append(chunk_file)
            return self._cache[chunk_file]
        
        data = np.load(self.dir / chunk_file, allow_pickle=True)
        chunk = {'embeddings': data['embeddings'], 'logits': data['logits']}
        
        self._cache[chunk_file] = chunk
        self._cache_order.append(chunk_file)
        
        while len(self._cache) > self.cache_size:
            del self._cache[self._cache_order.pop(0)]
        
        return chunk
    
    def get(self, path):
        if path not in self.path_to_loc:
            return None, None
        chunk_file, idx = self.path_to_loc[path]
        chunk = self._load_chunk(chunk_file)
        return (
            torch.from_numpy(chunk['embeddings'][idx].astype(np.float32)),
            torch.from_numpy(chunk['logits'][idx].astype(np.float32))
        )
    
    def __contains__(self, path):
        return path in self.path_to_loc


class PrescriptionDataset(Dataset):
    """Dataset for verifier training."""
    
    def __init__(self, store, prescriptions, error_rate=0.5, is_training=True):
        self.store = store
        self.prescriptions = prescriptions
        self.error_rate = error_rate
        self.is_training = is_training
        self.num_classes = store.num_classes
        self.class_to_idx = store.class_to_idx
        
        if is_training and error_rate > 0:
            self.pill_lib = defaultdict(list)
            for rx in prescriptions:
                for pill in rx['pills']:
                    if pill['path'] in store:
                        self.pill_lib[pill['ndc']].append(pill['path'])
            self.pill_lib = dict(self.pill_lib)
            self.ndc_list = list(self.pill_lib.keys())
        else:
            self.pill_lib = {}
            self.ndc_list = []
    
    def __len__(self):
        return len(self.prescriptions)
    
    def __getitem__(self, idx):
        rx = self.prescriptions[idx]
        pills = rx['pills']
        
        # Expected composition
        expected = torch.zeros(self.num_classes)
        for p in pills:
            expected[self.class_to_idx.get(p['ndc'], 0)] += 1
        expected = expected / len(pills)
        
        # Setup
        is_correct = True
        pill_correct = [1.0] * len(pills)
        paths = [p['path'] for p in pills]
        
        # Error injection
        if self.is_training and self.ndc_list and random.random() < self.error_rate:
            is_correct = False
            for err_idx in random.sample(range(len(pills)), min(3, len(pills))):
                orig = pills[err_idx]['ndc']
                cands = [n for n in self.ndc_list if n != orig]
                if cands:
                    wrong = random.choice(cands)
                    if self.pill_lib.get(wrong):
                        paths[err_idx] = random.choice(self.pill_lib[wrong])
                        pill_correct[err_idx] = 0.0
        
        # Get embeddings
        embs, logs = [], []
        for path in paths:
            e, l = self.store.get(path)
            embs.append(e if e is not None else torch.zeros(EMBED_DIM))
            logs.append(l if l is not None else torch.zeros(self.num_classes))
        
        return {
            'embeddings': torch.stack(embs),
            'logits': torch.stack(logs),
            'expected': expected,
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': torch.tensor(pill_correct),
            'num_pills': len(pills)
        }


def collate_images(batch):
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        [b[2] for b in batch]
    )


def collate_prescriptions(batch):
    B = len(batch)
    max_pills = max(b['num_pills'] for b in batch)
    num_classes = batch[0]['logits'].size(-1)
    
    emb = torch.zeros(B, max_pills, EMBED_DIM)
    log = torch.zeros(B, max_pills, num_classes)
    mask = torch.ones(B, max_pills, dtype=torch.bool)
    pill_correct = torch.zeros(B, max_pills)
    is_correct = torch.zeros(B)
    expected = torch.zeros(B, num_classes)
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        emb[i, :n] = b['embeddings']
        log[i, :n] = b['logits']
        mask[i, :n] = False
        pill_correct[i, :n] = b['pill_correct']
        is_correct[i] = b['is_correct']
        expected[i] = b['expected']
    
    return {
        'embeddings': emb, 'logits': log, 'mask': mask,
        'pill_correct': pill_correct, 'is_correct': is_correct, 'expected': expected
    }


# =============================================================================
# Extraction
# =============================================================================

def run_extraction(args, is_distributed, rank, local_rank, world_size, device, killer):
    """Extract embeddings from all images."""
    
    print_rank0("\n" + "=" * 60, rank)
    print_rank0("PHASE 2A: EMBEDDING EXTRACTION", rank)
    print_rank0("=" * 60, rank)
    
    emb_dir = Path(args.output_dir) / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    index_file = emb_dir / 'index.json'
    if index_file.exists() and not args.force_extract:
        print_rank0("  Embeddings already exist. Use --force-extract to redo.", rank)
        return True
    
    # Load backbone
    print_rank0("\nLoading backbone...", rank)
    ckpt = torch.load(args.backbone, map_location=device)
    num_classes = ckpt['num_classes']
    class_to_idx = ckpt['class_to_idx']
    
    backbone = PillClassifier(num_classes).to(device)
    backbone.load_state_dict(ckpt['model'])
    backbone.eval()
    
    if is_distributed:
        backbone = DDP(backbone, device_ids=[local_rank])
    
    # Load paths
    print_rank0("\nLoading image paths...", rank)
    data_index = os.path.join(args.data_dir, 'dataset_index.csv')
    paths = []
    seen = set()
    with open(data_index) as f:
        for row in csv.DictReader(f):
            p = row['relative_path']
            if p not in seen:
                seen.add(p)
                paths.append(p)
    
    print_rank0(f"  Total images: {len(paths):,}", rank)
    print_rank0(f"  Per GPU: ~{len(paths) // world_size:,}", rank)
    
    # Check for resume
    progress_file = emb_dir / f'progress_{rank}.json'
    processed = set()
    chunk_files = []
    chunk_id = 0
    
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
        processed = set(prog['processed'])
        chunk_files = prog['chunks']
        chunk_id = len(chunk_files)
        print_rank0(f"  Resuming: {len(processed):,} already done", rank)
    
    # Dataset
    dataset = ImageDataset(args.data_dir, paths, args.input_size)
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_images,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0
    )
    
    # Extract
    emb_buf, log_buf, path_buf = [], [], []
    total = 0
    failed = 0
    
    pbar = tqdm(loader, desc=f'GPU {rank}', disable=rank != 0)
    
    with torch.no_grad():
        for images, indices, valid in pbar:
            if killer.kill_now:
                break
            
            images = images.to(device, non_blocking=True)
            
            with autocast():
                model = backbone.module if is_distributed else backbone
                emb, log = model.get_embeddings_and_logits(images)
            
            emb = emb.cpu().numpy()
            log = log.cpu().numpy()
            
            for i, (idx, ok) in enumerate(zip(indices, valid)):
                if idx in processed:
                    continue
                if ok:
                    emb_buf.append(emb[i])
                    log_buf.append(log[i])
                    path_buf.append(paths[idx])
                    total += 1
                else:
                    failed += 1
                processed.add(idx)
            
            # Save chunk
            if len(emb_buf) >= CHUNK_SIZE:
                chunk_name = f'chunk_{rank}_{chunk_id}.npz'
                np.savez_compressed(
                    emb_dir / chunk_name,
                    embeddings=np.stack(emb_buf).astype(np.float16),
                    logits=np.stack(log_buf).astype(np.float16),
                    paths=np.array(path_buf, dtype=object)
                )
                chunk_files.append(chunk_name)
                
                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump({'processed': list(processed), 'chunks': chunk_files}, f)
                
                if rank == 0:
                    pbar.write(f"  Saved {chunk_name}")
                
                emb_buf, log_buf, path_buf = [], [], []
                chunk_id += 1
                gc.collect()
            
            if rank == 0:
                pbar.set_postfix(done=total, fail=failed, chunks=len(chunk_files))
    
    # Save remaining
    if emb_buf:
        chunk_name = f'chunk_{rank}_{chunk_id}.npz'
        np.savez_compressed(
            emb_dir / chunk_name,
            embeddings=np.stack(emb_buf).astype(np.float16),
            logits=np.stack(log_buf).astype(np.float16),
            paths=np.array(path_buf, dtype=object)
        )
        chunk_files.append(chunk_name)
        with open(progress_file, 'w') as f:
            json.dump({'processed': list(processed), 'chunks': chunk_files}, f)
    
    print(f"GPU {rank}: {total:,} extracted, {failed} failed, {len(chunk_files)} chunks")
    
    safe_barrier()
    
    # Rank 0 creates index
    if rank == 0:
        print("\nCreating index...")
        all_chunks = []
        total_imgs = 0
        
        for r in range(world_size):
            pf = emb_dir / f'progress_{r}.json'
            if pf.exists():
                with open(pf) as f:
                    prog = json.load(f)
                all_chunks.extend(prog['chunks'])
                total_imgs += len([p for p in prog['processed'] if isinstance(p, int)])
        
        with open(index_file, 'w') as f:
            json.dump({
                'chunk_files': all_chunks,
                'total_images': total_imgs,
                'num_classes': num_classes,
                'class_to_idx': class_to_idx,
                'embed_dim': EMBED_DIM
            }, f, indent=2)
        
        # Cleanup progress files
        if not killer.kill_now:
            for r in range(world_size):
                pf = emb_dir / f'progress_{r}.json'
                if pf.exists():
                    pf.unlink()
        
        print(f"  â Index created: {len(all_chunks)} chunks")
    
    safe_barrier()
    return not killer.kill_now


# =============================================================================
# Training
# =============================================================================

def load_prescriptions(index_file, split, min_pills=5, max_pills=200):
    """Load prescriptions from index."""
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


def run_training(args, is_distributed, rank, local_rank, world_size, device, killer):
    """Train the verifier."""
    
    print_rank0("\n" + "=" * 60, rank)
    print_rank0("PHASE 2B: VERIFIER TRAINING", rank)
    print_rank0("=" * 60, rank)
    
    emb_dir = Path(args.output_dir) / 'embeddings'
    
    # Check embeddings exist
    if not (emb_dir / 'index.json').exists():
        print_rank0("  ERROR: Embeddings not found. Run 'extract' first.", rank)
        return False
    
    # Load backbone info
    ckpt = torch.load(args.backbone, map_location='cpu')
    num_classes = ckpt['num_classes']
    
    # Load embeddings
    print_rank0("\nLoading embeddings...", rank)
    store = EmbeddingStore(emb_dir, cache_size=30)
    print_rank0(f"  Indexed {len(store.path_to_loc):,} embeddings", rank)
    
    # Load prescriptions
    print_rank0("\nLoading prescriptions...", rank)
    data_index = os.path.join(args.data_dir, 'dataset_index.csv')
    train_rx = load_prescriptions(data_index, 'train')
    val_rx = load_prescriptions(data_index, 'valid')
    print_rank0(f"  Train: {len(train_rx):,}", rank)
    print_rank0(f"  Val: {len(val_rx):,}", rank)
    
    # Datasets
    train_ds = PrescriptionDataset(store, train_rx, args.error_rate, True)
    val_ds = PrescriptionDataset(store, val_rx, args.error_rate, False)
    
    train_sampler = DistributedSampler(train_ds) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_ds, args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
        collate_fn=collate_prescriptions, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, sampler=val_sampler, shuffle=False,
        collate_fn=collate_prescriptions, num_workers=4, pin_memory=True
    )
    
    # Model
    model = PrescriptionVerifier(num_classes, args.hidden_dim, 8, args.num_layers).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    print_rank0(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}", rank)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Resume
    start_epoch, start_batch, best_acc = 0, 0, 0.0
    ckpt_path = Path(args.output_dir) / 'verifier_latest.pt'
    
    if ckpt_path.exists():
        print_rank0(f"\nResuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location=device)
        
        state = ckpt['model']
        if is_distributed:
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        start_batch = ckpt.get('batch', 0)
        best_acc = ckpt.get('best_acc', 0)
        
        for _ in range(start_epoch):
            scheduler.step()
        
        print_rank0(f"  Epoch {start_epoch}, batch {start_batch}, best={best_acc:.4f}", rank)
    
    # Training loop
    print_rank0("\nStarting training...", rank)
    
    for epoch in range(start_epoch, args.epochs):
        if killer.kill_now:
            break
        
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        total_anom_correct, total_anom_count = 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=rank != 0)
        
        for batch_idx, batch in enumerate(pbar):
            if killer.kill_now:
                if rank == 0:
                    torch.save({
                        'epoch': epoch, 'batch': batch_idx,
                        'model': model.module.state_dict() if is_distributed else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc
                    }, ckpt_path)
                break
            
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            
            emb = batch['embeddings'].to(device)
            log = batch['logits'].to(device)
            mask = batch['mask'].to(device)
            is_correct = batch['is_correct'].to(device)
            pill_correct = batch['pill_correct'].to(device)
            expected = batch['expected'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                script_logit, anom_logits = model(emb, log, expected, mask)
                
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
                
                loss = script_loss + 2.0 * anom_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            B = emb.size(0)
            total_loss += loss.item() * B
            
            with torch.no_grad():
                pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
                total_correct += (pred == is_correct).sum().item()
                total_count += B
                
                if valid.any():
                    anom_pred = (torch.sigmoid(anom_logits) > 0.5).float()
                    total_anom_correct += ((anom_pred == (1 - pill_correct)) & valid).sum().item()
                    total_anom_count += valid.sum().item()
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/total_count:.4f}',
                    'script': f'{total_correct/total_count:.4f}',
                    'anom': f'{total_anom_correct/max(1,total_anom_count):.4f}'
                })
            
            # Periodic save
            if rank == 0 and args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
                torch.save({
                    'epoch': epoch, 'batch': batch_idx + 1,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }, ckpt_path)
        
        if killer.kill_now:
            break
        
        # Reset start_batch
        start_batch = 0
        
        # Validation
        model.eval()
        val_correct, val_count = 0, 0
        val_anom_correct, val_anom_count = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                emb = batch['embeddings'].to(device)
                log = batch['logits'].to(device)
                mask = batch['mask'].to(device)
                is_correct = batch['is_correct'].to(device)
                pill_correct = batch['pill_correct'].to(device)
                expected = batch['expected'].to(device)
                
                with autocast():
                    script_logit, anom_logits = model(emb, log, expected, mask)
                
                pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
                val_correct += (pred == is_correct).sum().item()
                val_count += emb.size(0)
                
                valid = ~mask
                if valid.any():
                    anom_pred = (torch.sigmoid(anom_logits) > 0.5).float()
                    val_anom_correct += ((anom_pred == (1 - pill_correct)) & valid).sum().item()
                    val_anom_count += valid.sum().item()
        
        # Sync metrics
        if is_distributed:
            stats = torch.tensor([val_correct, val_count, val_anom_correct, val_anom_count], 
                               device=device, dtype=torch.float64)
            safe_all_reduce(stats)
            val_correct, val_count, val_anom_correct, val_anom_count = stats.tolist()
        
        val_acc = val_correct / max(1, val_count)
        val_anom_acc = val_anom_correct / max(1, val_anom_count)
        
        scheduler.step()
        
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train: loss={total_loss/total_count:.4f}, script={total_correct/total_count:.4f}")
            print(f"  Val:   script={val_acc:.4f}, anomaly={val_anom_acc:.4f}")
            
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                print("  â New best!")
            
            torch.save({
                'epoch': epoch + 1, 'batch': 0,
                'model': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, ckpt_path)
            
            if is_best:
                torch.save({
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'best_acc': best_acc
                }, Path(args.output_dir) / 'verifier_best.pt')
    
    # Save final
    if rank == 0:
        print("\nSaving final model...")
        backbone_ckpt = torch.load(args.backbone, map_location='cpu')
        torch.save({
            'backbone': backbone_ckpt['model'],
            'verifier': model.module.state_dict() if is_distributed else model.state_dict(),
            'num_classes': num_classes,
            'class_to_idx': backbone_ckpt['class_to_idx'],
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers
        }, Path(args.output_dir) / 'full_model.pt')
        
        print("  â Saved full_model.pt")
    
    return not killer.kill_now


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data-dir', type=str, required=True)
    common.add_argument('--backbone', type=str, required=True)
    common.add_argument('--output-dir', type=str, default='./output_phase2')
    common.add_argument('--input-size', type=int, default=224)
    
    # Extract command
    p_extract = subparsers.add_parser('extract', parents=[common])
    p_extract.add_argument('--batch-size', type=int, default=256)
    p_extract.add_argument('--num-workers', type=int, default=8)
    p_extract.add_argument('--force-extract', action='store_true')
    
    # Train command
    p_train = subparsers.add_parser('train', parents=[common])
    p_train.add_argument('--batch-size', type=int, default=32)
    p_train.add_argument('--epochs', type=int, default=20)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--hidden-dim', type=int, default=256)
    p_train.add_argument('--num-layers', type=int, default=4)
    p_train.add_argument('--error-rate', type=float, default=0.5)
    p_train.add_argument('--save-every', type=int, default=100)
    
    # All command
    p_all = subparsers.add_parser('all', parents=[common])
    p_all.add_argument('--extract-batch-size', type=int, default=256)
    p_all.add_argument('--train-batch-size', type=int, default=32)
    p_all.add_argument('--num-workers', type=int, default=8)
    p_all.add_argument('--epochs', type=int, default=20)
    p_all.add_argument('--lr', type=float, default=1e-4)
    p_all.add_argument('--hidden-dim', type=int, default=256)
    p_all.add_argument('--num-layers', type=int, default=4)
    p_all.add_argument('--error-rate', type=float, default=0.5)
    p_all.add_argument('--save-every', type=int, default=100)
    p_all.add_argument('--force-extract', action='store_true')
    
    args = parser.parse_args()
    
    killer = GracefulKiller()
    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    
    print_rank0("=" * 60, rank)
    print_rank0("PHASE 2: CONSOLIDATED PIPELINE", rank)
    print_rank0("=" * 60, rank)
    print_rank0(f"Command: {args.command}", rank)
    print_rank0(f"World size: {world_size}", rank)
    print_rank0(f"Device: {device}", rank)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    if args.command == 'extract':
        success = run_extraction(args, is_distributed, rank, local_rank, world_size, device, killer)
    
    elif args.command == 'train':
        success = run_training(args, is_distributed, rank, local_rank, world_size, device, killer)
    
    elif args.command == 'all':
        # Run extraction
        args.batch_size = args.extract_batch_size
        args.num_workers = args.num_workers
        success = run_extraction(args, is_distributed, rank, local_rank, world_size, device, killer)
        
        if success and not killer.kill_now:
            safe_barrier()
            # Run training
            args.batch_size = args.train_batch_size
            success = run_training(args, is_distributed, rank, local_rank, world_size, device, killer)
    
    if is_distributed:
        dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "=" * 60)
        if success and not killer.kill_now:
            print("â COMPLETE!")
        else:
            print("â  INTERRUPTED - Run again to resume")
        print("=" * 60)


if __name__ == '__main__':
    main()
