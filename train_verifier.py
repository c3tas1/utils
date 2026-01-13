#!/usr/bin/env python3
"""
Phase 2: Verifier Training (Robust + Prescription-Aware)
========================================================

Key improvements:
1. PRESCRIPTION-AWARE: Verifier knows expected NDCs for each prescription
2. ROBUST: Handles NCCL timeouts, socket errors, auto-saves frequently
3. DDP SUPPORT: Multi-GPU with proper synchronization

Usage:
    # Single GPU
    python train_phase2_verifier.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt
    
    # Multi-GPU
    torchrun --nproc_per_node=8 train_phase2_verifier.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt
    
    # Resume after crash
    torchrun --nproc_per_node=8 train_phase2_verifier.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --resume output_phase2/verifier_latest.pt \
        --skip-extraction
"""

import os
import sys
import csv
import argparse
import random
import pickle
import signal
import time
import socket
from pathlib import Path
from datetime import timedelta
from collections import defaultdict, Counter
from contextlib import contextmanager

# ============================================================================
# NCCL ROBUSTNESS: Set environment variables BEFORE importing torch
# ============================================================================
os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 min timeout
os.environ.setdefault('NCCL_IB_TIMEOUT', '24')  # InfiniBand timeout
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0,en0,eno1,enp')  # Network interface
os.environ.setdefault('NCCL_DEBUG', 'WARN')  # Show warnings
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Async error handling
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')  # Blocking wait
os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # Enable P2P (try '1' if issues)
os.environ.setdefault('NCCL_SHM_DISABLE', '0')  # Enable shared memory

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

def init_distributed_robust():
    """Initialize distributed training with multiple fallback strategies."""
    
    if 'RANK' not in os.environ:
        return False, 0, 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Strategy 1: Try NCCL with long timeout
    backends_to_try = [
        ('nccl', timedelta(minutes=60)),
        ('nccl', timedelta(minutes=30)),
        ('gloo', timedelta(minutes=30)),
    ]
    
    for backend, timeout in backends_to_try:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            
            if rank == 0:
                print(f"  Trying {backend} backend with {timeout} timeout...")
            
            dist.init_process_group(
                backend=backend,
                timeout=timeout,
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Test communication
            test_tensor = torch.ones(1, device=device)
            dist.all_reduce(test_tensor)
            
            if rank == 0:
                print(f"  â Successfully initialized with {backend}")
            
            return True, rank, local_rank, world_size, device
            
        except Exception as e:
            if rank == 0:
                print(f"  â {backend} failed: {e}")
            continue
    
    # All strategies failed - fall back to single GPU
    if rank == 0:
        print("  â  All distributed backends failed. Running single GPU.")
    
    return False, 0, 0, 1, device


# =============================================================================
# Robustness Utilities
# =============================================================================

class GracefulKiller:
    """Handle SIGTERM/SIGINT gracefully."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        print("\nâ ï¸  Received shutdown signal. Saving checkpoint...")
        self.kill_now = True


def safe_all_reduce(tensor, op=dist.ReduceOp.SUM, max_retries=3):
    """All-reduce with retry on failure."""
    for attempt in range(max_retries):
        try:
            dist.all_reduce(tensor, op=op)
            return True
        except Exception as e:
            print(f"  Warning: all_reduce failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return False


def safe_barrier(max_retries=3):
    """Barrier with retry on failure."""
    if not dist.is_initialized():
        return True
    
    for attempt in range(max_retries):
        try:
            dist.barrier()
            return True
        except Exception as e:
            print(f"  Warning: barrier failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return False


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
# Prescription-Aware Verifier
# =============================================================================

class PrescriptionAwareVerifier(nn.Module):
    """
    Verifier that uses BOTH:
    1. Pill embeddings (what the model sees)
    2. Expected prescription composition (what SHOULD be there)
    
    This allows the model to compare:
    - "I see 10 round white pills" vs "Prescription says 10x Aspirin"
    """
    
    def __init__(self, num_classes: int, embed_dim: int = 512, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # NDC embedding (learnable embedding for each drug class)
        self.ndc_embedding = nn.Embedding(num_classes, hidden_dim)
        
        # Pill feature projection: embedding + confidence + entropy + predicted_class_embed
        # 512 (embedding) + 1 (conf) + 1 (entropy) + hidden_dim (predicted NDC embed)
        self.pill_projector = nn.Sequential(
            nn.Linear(embed_dim + 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prescription composition encoder
        # Encodes "expected" pill counts per NDC
        self.rx_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention: pills attend to prescription expectation
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Transformer encoder for pill sequence
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
        
        # Summary token for prescription-level prediction
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Output heads
        self.script_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat summary + rx_encoding
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.anomaly_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, pill_embeddings, pill_logits, expected_composition, mask):
        """
        Args:
            pill_embeddings: (B, N, 512) - backbone embeddings
            pill_logits: (B, N, num_classes) - classification logits
            expected_composition: (B, num_classes) - count of each NDC in prescription
            mask: (B, N) - True for padding positions
        """
        B, N, _ = pill_embeddings.shape
        device = pill_embeddings.device
        
        # Compute pill features
        probs = F.softmax(pill_logits, dim=-1)
        confidences = probs.max(dim=-1).values.unsqueeze(-1)  # (B, N, 1)
        entropies = -(probs * (probs + 1e-8).log()).sum(dim=-1).unsqueeze(-1)
        entropies = entropies / np.log(self.num_classes)  # Normalize
        
        # Get predicted class embeddings
        predicted_classes = pill_logits.argmax(dim=-1)  # (B, N)
        predicted_ndc_embeds = self.ndc_embedding(predicted_classes)  # (B, N, hidden_dim)
        
        # Combine pill features
        pill_features = torch.cat([
            pill_embeddings, confidences, entropies, predicted_ndc_embeds
        ], dim=-1)
        pill_tokens = self.pill_projector(pill_features)  # (B, N, hidden_dim)
        
        # Encode expected prescription composition
        rx_encoding = self.rx_encoder(expected_composition)  # (B, hidden_dim)
        rx_tokens = rx_encoding.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Cross-attention: pills attend to prescription expectation
        # "Does what I see match what's expected?"
        pill_tokens_attended, _ = self.cross_attention(
            pill_tokens, rx_tokens, rx_tokens,
            key_padding_mask=None  # rx_tokens has no padding
        )
        pill_tokens = pill_tokens + pill_tokens_attended  # Residual
        
        # Add summary token
        summary = self.summary_token.expand(B, -1, -1)
        tokens = torch.cat([summary, pill_tokens], dim=1)  # (B, 1+N, hidden)
        
        # Extend mask for summary token
        summary_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([summary_mask, mask], dim=1)
        
        # Transformer encoding
        encoded = self.encoder(tokens, src_key_padding_mask=full_mask)
        
        # Extract outputs
        summary_out = encoded[:, 0, :]  # (B, hidden)
        pill_out = encoded[:, 1:, :]  # (B, N, hidden)
        
        # Script prediction: combine summary with rx_encoding
        script_features = torch.cat([summary_out, rx_encoding], dim=-1)
        script_logit = self.script_head(script_features)  # (B, 1)
        
        # Anomaly prediction per pill
        anomaly_logits = self.anomaly_head(pill_out).squeeze(-1)  # (B, N)
        
        return script_logit, anomaly_logits


# =============================================================================
# Dataset with Prescription Composition
# =============================================================================

class PrecomputedPrescriptionDataset(Dataset):
    """
    Dataset with precomputed embeddings AND expected composition.
    """
    
    def __init__(self, embeddings_file: str, prescriptions: list, 
                 num_classes: int, error_rate: float = 0.5, is_training: bool = True):
        
        # Load precomputed embeddings
        print(f"Loading embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.logits = data['logits']
        self.class_to_idx = data['class_to_idx']
        self.num_classes = num_classes
        
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
        
        print(f"  Loaded {len(prescriptions)} prescriptions")
    
    def __len__(self):
        return len(self.prescriptions)
    
    def __getitem__(self, idx):
        rx = self.prescriptions[idx]
        pills = rx['pills']
        
        # Compute EXPECTED composition (before any error injection)
        expected_composition = torch.zeros(self.num_classes)
        for pill in pills:
            ndc_idx = self.class_to_idx[pill['ndc']]
            expected_composition[ndc_idx] += 1
        # Normalize by total pills
        expected_composition = expected_composition / len(pills)
        
        # Prepare pill data
        is_correct = True
        pill_correct = [1.0] * len(pills)
        pill_paths = [p['path'] for p in pills]
        pill_labels = [self.class_to_idx[p['ndc']] for p in pills]
        
        # Error injection (training only)
        if self.is_training and random.random() < self.error_rate:
            is_correct = False
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
                embeddings.append(torch.zeros(512))
                logits_list.append(torch.zeros(self.num_classes))
        
        embeddings = torch.stack(embeddings)
        logits = torch.stack(logits_list)
        
        return {
            'embeddings': embeddings,
            'logits': logits,
            'labels': torch.tensor(pill_labels, dtype=torch.long),
            'expected_composition': expected_composition,
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
    labels = torch.full((B, max_pills), -100, dtype=torch.long)
    mask = torch.ones(B, max_pills, dtype=torch.bool)
    pill_correct = torch.zeros(B, max_pills)
    is_correct = torch.zeros(B)
    expected_composition = torch.zeros(B, num_classes)
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        embeddings[i, :n] = b['embeddings']
        logits[i, :n] = b['logits']
        labels[i, :n] = b['labels']
        mask[i, :n] = False
        pill_correct[i, :n] = b['pill_correct']
        is_correct[i] = b['is_correct']
        expected_composition[i] = b['expected_composition']
    
    return {
        'embeddings': embeddings,
        'logits': logits,
        'labels': labels,
        'mask': mask,
        'pill_correct': pill_correct,
        'is_correct': is_correct,
        'expected_composition': expected_composition
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
    
    samples = list(set(samples))
    print(f"  Total unique images: {len(samples)}")
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings = {}
    logits = {}
    failed = []
    
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
                failed.append(path)
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
    
    if failed:
        print(f"  Warning: Failed to load {len(failed)} images")
    
    # Get class_to_idx
    all_ndcs = set()
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ndcs.add(row['ndc'])
    class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
    
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
# Training with Robustness
# =============================================================================

def train_epoch_robust(verifier, loader, optimizer, scaler, device, epoch, rank, killer, args, is_distributed, start_batch=0):
    """Training with batch-level checkpointing for crash recovery."""
    verifier.train()
    
    total_loss = 0
    total_script_correct = 0
    total_script_count = 0
    total_anomaly_correct = 0
    total_anomaly_count = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f'Epoch {epoch}', initial=start_batch, total=len(loader))
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        # Skip batches if resuming mid-epoch
        if batch_idx < start_batch:
            continue
        
        # Check for shutdown signal
        if killer.kill_now:
            if rank == 0:
                save_checkpoint(verifier, optimizer, epoch, batch_idx,
                              {'script_acc': 0}, args, is_distributed)
            return None
        
        embeddings = batch['embeddings'].to(device, non_blocking=True)
        logits = batch['logits'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        is_correct = batch['is_correct'].to(device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(device, non_blocking=True)
        expected_composition = batch['expected_composition'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        try:
            with autocast():
                script_logit, anomaly_logits = verifier(
                    embeddings, logits, expected_composition, mask
                )
                
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
        except RuntimeError as e:
            if 'NCCL' in str(e) or 'socket' in str(e).lower():
                # Critical error - save and exit
                if rank == 0:
                    print(f"\nâ ï¸  NCCL/Socket error at batch {batch_idx}: {e}")
                    save_checkpoint(verifier, optimizer, epoch, batch_idx,
                                  {'script_acc': 0}, args, is_distributed)
                return None
            else:
                # Non-critical error - skip batch
                print(f"  Warning: Error in batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue
        
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
                'loss': f'{total_loss/max(1,total_script_count):.4f}',
                'script': f'{total_script_correct/max(1,total_script_count):.4f}',
                'anomaly': f'{total_anomaly_correct/max(1,total_anomaly_count):.4f}'
            })
        
        # Periodic checkpoint (for crash recovery)
        if rank == 0 and args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
            save_checkpoint(verifier, optimizer, epoch, batch_idx + 1,
                          {'script_acc': total_script_correct/max(1,total_script_count)}, 
                          args, is_distributed)
    
    return {
        'loss': total_loss / max(1, total_script_count),
        'script_acc': total_script_correct / max(1, total_script_count),
        'anomaly_acc': total_anomaly_correct / max(1, total_anomaly_count)
    }


@torch.no_grad()
def validate(verifier, loader, device, rank=0, is_distributed=False):
    verifier.eval()
    
    total_script_correct = 0
    total_script_count = 0
    total_anomaly_correct = 0
    total_anomaly_count = 0
    
    for batch in loader:
        embeddings = batch['embeddings'].to(device)
        logits = batch['logits'].to(device)
        mask = batch['mask'].to(device)
        is_correct = batch['is_correct'].to(device)
        pill_correct = batch['pill_correct'].to(device)
        expected_composition = batch['expected_composition'].to(device)
        
        with autocast():
            script_logit, anomaly_logits = verifier(
                embeddings, logits, expected_composition, mask
            )
        
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
    
    # Sync across GPUs
    if is_distributed and dist.is_initialized():
        stats = torch.tensor([
            total_script_correct, total_script_count,
            total_anomaly_correct, total_anomaly_count
        ], device=device, dtype=torch.float64)
        
        if not safe_all_reduce(stats):
            print("  Warning: Failed to sync validation stats")
        else:
            total_script_correct, total_script_count, \
            total_anomaly_correct, total_anomaly_count = stats.tolist()
    
    return {
        'script_acc': total_script_correct / max(1, total_script_count),
        'anomaly_acc': total_anomaly_correct / max(1, total_anomaly_count)
    }


def save_checkpoint(verifier, optimizer, epoch, batch, metrics, args, is_distributed, is_best=False):
    """Save checkpoint with all necessary info for resume."""
    
    state = {
        'epoch': epoch,
        'batch': batch,
        'verifier': verifier.module.state_dict() if is_distributed else verifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'script_acc': metrics.get('script_acc', 0),
        'anomaly_acc': metrics.get('anomaly_acc', 0),
        'args': vars(args)
    }
    
    # Always save latest (overwrites)
    torch.save(state, os.path.join(args.output_dir, 'verifier_latest.pt'))
    
    # Save best
    if is_best:
        torch.save(state, os.path.join(args.output_dir, 'verifier_best.pt'))
    
    # Periodic saves (every 5 epochs)
    if batch == 0 and (epoch + 1) % 5 == 0:
        torch.save(state, os.path.join(args.output_dir, f'verifier_epoch_{epoch}.pt'))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Verifier Training (Robust)')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output_phase2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--error-rate', type=float, default=0.5)
    parser.add_argument('--skip-extraction', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=100, 
                       help='Save checkpoint every N batches (for crash recovery)')
    args = parser.parse_args()
    
    # Graceful shutdown handler
    killer = GracefulKiller()
    
    # Robust distributed initialization
    is_distributed, rank, local_rank, world_size, device = init_distributed_robust()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        print("=" * 60)
        print("PHASE 2: PRESCRIPTION-AWARE VERIFIER (ROBUST)")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size} Ã {world_size} = {args.batch_size * world_size}")
        print(f"Save every: {args.save_every} batches")
    
    # Load backbone
    if rank == 0:
        print("\nLoading backbone...")
    
    checkpoint = torch.load(args.backbone, map_location=device)
    num_classes = checkpoint['num_classes']
    class_to_idx = checkpoint['class_to_idx']
    
    backbone = PillClassifier(num_classes).to(device)
    backbone.load_state_dict(checkpoint['model'])
    backbone.eval()
    
    if rank == 0:
        print(f"  Classes: {num_classes}")
        print(f"  Backbone accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    # Extract embeddings (rank 0 only)
    index_file = os.path.join(args.data_dir, 'dataset_index.csv')
    embeddings_file = os.path.join(args.output_dir, 'embeddings_cache.pkl')
    
    if rank == 0:
        if not args.skip_extraction or not os.path.exists(embeddings_file):
            extract_embeddings(
                backbone, args.data_dir, index_file, embeddings_file,
                device, batch_size=64, input_size=args.input_size
            )
    
    safe_barrier()
    
    # Load prescriptions
    if rank == 0:
        print("\nLoading prescriptions...")
    
    train_prescriptions = load_prescriptions(index_file, 'train')
    val_prescriptions = load_prescriptions(index_file, 'valid')
    
    if rank == 0:
        print(f"  Train: {len(train_prescriptions)}")
        print(f"  Val: {len(val_prescriptions)}")
    
    # Datasets
    train_dataset = PrecomputedPrescriptionDataset(
        embeddings_file, train_prescriptions, num_classes,
        error_rate=args.error_rate, is_training=True
    )
    val_dataset = PrecomputedPrescriptionDataset(
        embeddings_file, val_prescriptions, num_classes,
        error_rate=args.error_rate, is_training=False
    )
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=collate_fn, num_workers=args.num_workers, 
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    verifier = PrescriptionAwareVerifier(
        num_classes=num_classes,
        embed_dim=512,
        hidden_dim=args.hidden_dim,
        num_heads=8,
        num_layers=args.num_layers,
        dropout=0.1
    ).to(device)
    
    if is_distributed:
        verifier = DDP(verifier, device_ids=[local_rank], find_unused_parameters=False)
    
    if rank == 0:
        params = sum(p.numel() for p in verifier.parameters())
        print(f"\nVerifier parameters: {params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()
    
    # Resume
    start_epoch = 0
    start_batch = 0
    best_script_acc = 0.0
    
    # Try to resume from latest checkpoint first
    resume_path = args.resume
    if resume_path is None:
        # Auto-detect latest checkpoint
        latest_path = os.path.join(args.output_dir, 'verifier_latest.pt')
        if os.path.exists(latest_path):
            resume_path = latest_path
            if rank == 0:
                print(f"\nAuto-detected checkpoint: {resume_path}")
    
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"\nResuming from {resume_path}")
        
        ckpt = torch.load(resume_path, map_location=device)
        
        if is_distributed:
            verifier.module.load_state_dict(ckpt['verifier'])
        else:
            verifier.load_state_dict(ckpt['verifier'])
        
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not load optimizer state: {e}")
        
        start_epoch = ckpt.get('epoch', 0)
        start_batch = ckpt.get('batch', 0)
        best_script_acc = ckpt.get('script_acc', 0.0)
        
        # If we completed the epoch, start from next
        if start_batch == 0 or start_batch >= len(train_loader) - 1:
            start_epoch += 1
            start_batch = 0
        
        for _ in range(start_epoch):
            scheduler.step()
        
        if rank == 0:
            print(f"  Resumed: epoch {start_epoch}, batch {start_batch}, best={best_script_acc:.4f}")
    
    # Training loop
    if rank == 0:
        print("\nStarting training...")
    
    for epoch in range(start_epoch, args.epochs):
        if killer.kill_now:
            break
        
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch_robust(
            verifier, train_loader, optimizer, scaler, 
            device, epoch, rank, killer, args,
            is_distributed, start_batch if epoch == start_epoch else 0
        )
        
        # Reset start_batch after first epoch
        start_batch = 0
        
        # Check for early exit
        if train_metrics is None:
            if rank == 0:
                print("\nSaving checkpoint before exit...")
                save_checkpoint(verifier, optimizer, epoch, len(train_loader),
                              {'script_acc': best_script_acc}, args, is_distributed)
            break
        
        val_metrics = validate(verifier, val_loader, device, rank, is_distributed)
        scheduler.step()
        
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Script: {train_metrics['script_acc']:.4f}, "
                  f"Anomaly: {train_metrics['anomaly_acc']:.4f}")
            print(f"  Val   - Script: {val_metrics['script_acc']:.4f}, "
                  f"Anomaly: {val_metrics['anomaly_acc']:.4f}")
            
            is_best = val_metrics['script_acc'] > best_script_acc
            if is_best:
                best_script_acc = val_metrics['script_acc']
                print(f"  â New best!")
            
            save_checkpoint(verifier, optimizer, epoch, 0, val_metrics, args, is_distributed, is_best)
    
    # Save final combined model
    if rank == 0:
        print("\nSaving combined model...")
        torch.save({
            'backbone': checkpoint['model'],
            'verifier': verifier.module.state_dict() if is_distributed else verifier.state_dict(),
            'num_classes': num_classes,
            'class_to_idx': class_to_idx,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers
        }, os.path.join(args.output_dir, 'full_model.pt'))
        
        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETE")
        print("=" * 60)
        print(f"Best script accuracy: {best_script_acc:.4f}")
    
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
