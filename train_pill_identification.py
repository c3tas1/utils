#!/usr/bin/env python3
"""
End-to-End Prescription-Aware Pill Verification Training (640x640 Optimized)
=============================================================================
Target Hardware: NVIDIA DGX A100 (8x A100 40GB GPUs)

Key Features:
1. Native 640x640 Resolution support (No resizing to 224).
2. SyncBatchNorm for stability at Batch Size = 2.
3. Gradient Checkpointing enabled to save VRAM.
4. Robust Image Loading (Skipping corrupt files).

Usage:
    # 1. Build Index
    python train_e2e_640px.py --data-dir /path/to/data --build-index

    # 2. Train (Multi-GPU)
    torchrun --nproc_per_node=8 train_e2e_640px.py \
        --data-dir /path/to/data \
        --output-dir ./output \
        --epochs 100
"""

import os
import sys
import re
import csv
import gc
import math
import random
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

# 3rd Party Imports
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
from torch.utils.checkpoint import checkpoint
from torchvision import transforms, models
from tqdm import tqdm

# Allow loading truncated images if necessary
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for end-to-end training."""
    
    # Data paths
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Model architecture
    num_classes: int = 1228  # Will be overwritten by index loader
    backbone: str = "resnet34"
    pill_embed_dim: int = 512
    verifier_hidden_dim: int = 256
    verifier_num_heads: int = 8
    verifier_num_layers: int = 4
    verifier_dropout: float = 0.1
    
    # Prescription constraints
    max_pills_per_prescription: int = 200
    min_pills_per_prescription: int = 5
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 2  # Small batch size (Requires SyncBatchNorm)
    gradient_accumulation_steps: int = 4
    
    # Learning rates (Differential)
    backbone_lr: float = 5e-5      # Lower for pretrained backbone
    classifier_lr: float = 1e-4    # Medium for head
    verifier_lr: float = 1e-4      # Higher for new transformer
    weight_decay: float = 0.01
    
    # LR schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Loss weights
    script_loss_weight: float = 1.0
    pill_anomaly_loss_weight: float = 1.0
    pill_classification_loss_weight: float = 1.0
    
    # Synthetic error injection
    error_rate: float = 0.5
    
    # Hardware / DDP
    use_amp: bool = True
    use_gradient_checkpointing: bool = True  # CRITICAL for 640x640 images
    max_grad_norm: float = 1.0
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = 30
    find_unused_parameters: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Logging & Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    log_every_n_steps: int = 10
    seed: int = 42
    resume_from: Optional[str] = None


# =============================================================================
# Infrastructure & Logging
# =============================================================================

def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    logger = logging.getLogger("PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers = []
    
    if rank == 0:
        # Console
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console)
        
        # File
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, f'train_{datetime.now():%Y%m%d_%H%M%S}.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
    
    return logger

class DDPManager:
    """Manages Distributed Data Parallel setup."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        self.device = torch.device('cuda:0')
    
    def setup(self) -> None:
        if 'RANK' not in os.environ:
            print("Running in single-GPU mode")
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            return
        
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.is_distributed = True
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # NCCL Optimization for A100
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        
        dist.init_process_group(
            backend=self.config.dist_backend,
            timeout=timedelta(minutes=self.config.dist_timeout_minutes),
            rank=self.rank,
            world_size=self.world_size
        )
        dist.barrier()
        if self.rank == 0:
            print(f"DDP Initialized: {self.world_size} GPUs")
    
    def cleanup(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    @property
    def is_main(self) -> bool:
        return self.rank == 0

@contextmanager
def ddp_sync_context(model: nn.Module, sync: bool = True):
    if isinstance(model, DDP) and not sync:
        with model.no_sync():
            yield
    else:
        yield


# =============================================================================
# Data Layer
# =============================================================================

class DatasetIndexer:
    """Fast CSV-based dataset indexing."""
    FILENAME_PATTERN = re.compile(r'^(.+)_(.+)_patch_(\d+)\.jpg$', re.IGNORECASE)
    
    @classmethod
    def build_index(cls, data_dir: str, output_file: str, splits: List[str] = ['train', 'valid']):
        print(f"Building index for {data_dir}...")
        stats = defaultdict(int)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'ndc', 'prescription_id', 'patch_no', 'relative_path'])
            
            for split in splits:
                split_dir = Path(data_dir) / split
                if not split_dir.exists(): continue
                
                ndc_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
                for ndc_dir in tqdm(ndc_dirs, desc=f"Indexing {split}"):
                    ndc = ndc_dir.name
                    for img_path in ndc_dir.glob('*.jpg'):
                        match = cls.FILENAME_PATTERN.match(img_path.name)
                        if match:
                            _, rx_id, patch_no = match.groups()
                            rel = str(img_path.relative_to(data_dir))
                            writer.writerow([split, ndc, rx_id, int(patch_no), rel])
                            stats[f'{split}_images'] += 1
                stats[f'{split}_ndcs'] = len(ndc_dirs)
        
        print("\nIndex Stats:")
        for k, v in stats.items(): print(f"  {k}: {v:,}")

    @classmethod
    def load_index(cls, index_file: str) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
        data_by_split = defaultdict(list)
        all_ndcs = set()
        
        with open(index_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_by_split[row['split']].append({
                    'ndc': row['ndc'],
                    'prescription_id': row['prescription_id'],
                    'patch_no': int(row['patch_no']),
                    'relative_path': row['relative_path']
                })
                all_ndcs.add(row['ndc'])
        
        class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
        return dict(data_by_split), class_to_idx


class PrescriptionDataset(Dataset):
    def __init__(self, data_dir: str, split_data: List[dict], class_to_idx: Dict[str, int], 
                 transform=None, max_pills=200, min_pills=5, error_rate=0.0, is_training=True):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform or self._default_transform(is_training)
        self.max_pills = max_pills
        self.min_pills = min_pills
        self.error_rate = error_rate
        self.is_training = is_training
        
        self.prescriptions = self._group_by_prescription(split_data)
        self.prescription_ids = list(self.prescriptions.keys())
        
        if error_rate > 0:
            self.pill_library = self._build_pill_library()
        else:
            self.pill_library = {}

    def _default_transform(self, is_training):
        # ImageNet normalization
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # TARGET RESOLUTION: 640x640
        if is_training:
            return transforms.Compose([
                transforms.Resize((672, 672)), # Slightly larger for random crop
                transforms.RandomCrop(640),    # Target size
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1),
                transforms.ToTensor(),
                norm
            ])
        else:
            return transforms.Compose([
                transforms.Resize(640),      # Resize smallest side to 640
                transforms.CenterCrop(640),  # Center crop 640x640
                transforms.ToTensor(),
                norm
            ])

    def _group_by_prescription(self, data):
        grouped = defaultdict(list)
        for x in data: grouped[x['prescription_id']].append(x)
        return {k: sorted(v, key=lambda i: i['patch_no']) for k, v in grouped.items() 
                if self.min_pills <= len(v) <= self.max_pills}

    def _build_pill_library(self):
        lib = defaultdict(list)
        for rx in self.prescriptions.values():
            for p in rx: lib[p['ndc']].append(p['relative_path'])
        return dict(lib)

    def _load_image(self, rel_path):
        try:
            img = Image.open(self.data_dir / rel_path).convert('RGB')
            return img
        except Exception as e:
            # Robust loading: Return black image if file corrupt
            # print(f"Error loading {rel_path}: {e}") # Uncomment for debug
            return Image.new('RGB', (640, 640), (0, 0, 0))

    def _inject_errors(self, pills):
        if not self.pill_library: return pills, []
        
        modified = [p.copy() for p in pills]
        n = len(pills)
        
        # Decide how many errors
        rtype = random.random()
        if rtype < 0.5: n_err = 1
        elif rtype < 0.85: n_err = random.randint(2, max(2, n // 4))
        else: n_err = random.randint(max(2, n // 4), max(2, n // 2))
        
        indices = random.sample(range(n), min(n_err, n))
        
        for i in indices:
            orig = modified[i]['ndc']
            candidates = [k for k in self.pill_library if k != orig]
            if candidates:
                wrong_ndc = random.choice(candidates)
                modified[i]['ndc'] = wrong_ndc  # Update label
                modified[i]['relative_path'] = random.choice(self.pill_library[wrong_ndc]) # Update image
                
        return modified, indices

    def __len__(self):
        return len(self.prescription_ids)

    def __getitem__(self, idx):
        rx_id = self.prescription_ids[idx]
        original_pills = self.prescriptions[rx_id]
        
        # Error injection logic
        inject = self.is_training and (random.random() < self.error_rate)
        if inject:
            pills, wrong_indices = self._inject_errors(original_pills)
            is_correct = False
        else:
            pills = original_pills
            wrong_indices = []
            is_correct = True
            
        images, labels = [], []
        for p in pills:
            img = self._load_image(p['relative_path'])
            if self.transform: img = self.transform(img)
            images.append(img)
            labels.append(self.class_to_idx[p['ndc']])
            
        # Expected composition (from ORIGINAL)
        comp = defaultdict(int)
        for p in original_pills: comp[p['ndc']] += 1
        exp_ndcs = [self.class_to_idx[k] for k in comp]
        exp_counts = list(comp.values())
        
        # Per-pill correctness mask (1=correct, 0=anomaly)
        pill_correct = torch.ones(len(pills), dtype=torch.float)
        for i in wrong_indices: pill_correct[i] = 0.0
        
        return {
            'images': torch.stack(images),
            'labels': torch.tensor(labels, dtype=torch.long),
            'expected_ndcs': torch.tensor(exp_ndcs, dtype=torch.long),
            'expected_counts': torch.tensor(exp_counts, dtype=torch.float),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': pill_correct,
            'num_pills': len(pills)
        }

def collate_fn(batch):
    """Pads variable length prescriptions into a batch."""
    max_p = max(b['num_pills'] for b in batch)
    max_e = max(len(b['expected_ndcs']) for b in batch)
    B = len(batch)
    C, H, W = batch[0]['images'].shape[1:]
    
    # Pre-allocate
    out = {
        'images': torch.zeros(B, max_p, C, H, W),
        'labels': torch.full((B, max_p), -100, dtype=torch.long),
        'pill_correct': torch.zeros(B, max_p),
        'pill_mask': torch.ones(B, max_p, dtype=torch.bool), # True = Padding
        
        'expected_ndcs': torch.zeros(B, max_e, dtype=torch.long),
        'expected_counts': torch.zeros(B, max_e, dtype=torch.float),
        'expected_mask': torch.ones(B, max_e, dtype=torch.bool),
        
        'is_correct': torch.zeros(B)
    }
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        ne = len(b['expected_ndcs'])
        
        out['images'][i, :n] = b['images']
        out['labels'][i, :n] = b['labels']
        out['pill_correct'][i, :n] = b['pill_correct']
        out['pill_mask'][i, :n] = False
        
        out['expected_ndcs'][i, :ne] = b['expected_ndcs']
        out['expected_counts'][i, :ne] = b['expected_counts']
        out['expected_mask'][i, :ne] = False
        
        out['is_correct'][i] = b['is_correct']
        
    return out


# =============================================================================
# Modeling
# =============================================================================

class ResNetBackbone(nn.Module):
    """ResNet backbone with Sequence (5D) support and Gradient Checkpointing."""
    def __init__(self, model_name: str, num_classes: int, use_checkpointing: bool):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        if model_name == "resnet34":
            w = models.ResNet34_Weights.IMAGENET1K_V1
            m = models.resnet34(weights=w)
            self.embed_dim = 512
        else:
            w = models.ResNet50_Weights.IMAGENET1K_V1
            m = models.resnet50(weights=w)
            self.embed_dim = 2048
            
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # Explicit dimensionality check
        is_sequence = (x.dim() == 5)
        
        if is_sequence:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            
        if self.use_checkpointing and self.training:
            x = checkpoint(self.stem, x, use_reentrant=False)
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
            
        x = self.avgpool(x).flatten(1)
        logits = self.classifier(x)
        
        if is_sequence:
            x = x.view(B, N, -1)
            logits = logits.view(B, N, -1)
            
        return x, logits

class PrescriptionVerifier(nn.Module):
    """Transformer-based Verification Head."""
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        dim = cfg.verifier_hidden_dim
        
        self.ndc_embed = nn.Embedding(cfg.num_classes, dim)
        
        # Prescription Encoder
        enc_layer = nn.TransformerEncoderLayer(dim, cfg.verifier_num_heads, dim*4, cfg.verifier_dropout, batch_first=True, norm_first=True)
        self.rx_enc = nn.TransformerEncoder(enc_layer, 2)
        
        # Feature Projector
        self.proj = nn.Sequential(
            nn.Linear(cfg.pill_embed_dim + 1, dim), # +1 for confidence score
            nn.LayerNorm(dim), nn.GELU(), nn.Dropout(cfg.verifier_dropout),
            nn.Linear(dim, dim)
        )
        
        # Cross Attention (Pills <-> Script)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, cfg.verifier_num_heads, dropout=cfg.verifier_dropout, batch_first=True)
            for _ in range(2)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        
        # Pill Encoder (Self-Attention)
        pill_layer = nn.TransformerEncoderLayer(dim, cfg.verifier_num_heads, dim*4, cfg.verifier_dropout, batch_first=True, norm_first=True)
        self.pill_enc = nn.TransformerEncoder(pill_layer, cfg.verifier_num_layers)
        
        # Output Heads
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(dim, cfg.verifier_num_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(dim)
        
        self.head_script = nn.Linear(dim, 1)
        self.head_anomaly = nn.Linear(dim, 1)

    def forward(self, embeddings, confidences, rx_ndcs, rx_counts, pill_mask, rx_mask):
        B = embeddings.size(0)
        
        # 1. Encode Prescription
        rx_emb = self.ndc_embed(rx_ndcs)
        # Add Count Signal: Normalize count and add as bias
        cnt_bias = (rx_counts / (rx_counts.sum(1, keepdim=True) + 1e-8)).unsqueeze(-1)
        rx_emb = rx_emb + (cnt_bias * 0.1)
        rx_feat = self.rx_enc(rx_emb, src_key_padding_mask=rx_mask)
        
        # 2. Project Pill Features
        pill_feat = self.proj(torch.cat([embeddings, confidences], dim=-1))
        
        # 3. Cross Attention
        for i in range(2):
            # Q=Pills, K=Script, V=Script
            attn, _ = self.cross_attn[i](pill_feat, rx_feat, rx_feat, key_padding_mask=rx_mask)
            pill_feat = self.cross_norms[i](pill_feat + attn)
            
        # 4. Self Attention
        pill_feat = self.pill_enc(pill_feat, src_key_padding_mask=pill_mask)
        
        # 5. Anomaly Output (Per pill)
        anomaly_logits = self.head_anomaly(pill_feat).squeeze(-1)
        
        # 6. Script Output (Global)
        token = self.pool_token.expand(B, -1, -1)
        pool, _ = self.pool_attn(token, pill_feat, pill_feat, key_padding_mask=pill_mask)
        script_logit = self.head_script(self.pool_norm(token + pool).squeeze(1))
        
        return script_logit, anomaly_logits

class EndToEndModel(nn.Module):
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.backbone = ResNetBackbone(cfg.backbone, cfg.num_classes, cfg.use_gradient_checkpointing)
        self.verifier = PrescriptionVerifier(cfg)
        
    def forward(self, img, rx_ndc, rx_cnt, p_mask, rx_mask):
        emb, logits = self.backbone(img)
        # Calculate Pill Confidence (Max Softmax)
        probs = F.softmax(logits, dim=-1)
        confs = probs.max(dim=-1).values.unsqueeze(-1)
        
        s_logit, a_logits = self.verifier(emb, confs, rx_ndc, rx_cnt, p_mask, rx_mask)
        return s_logit, a_logits, logits


# =============================================================================
# Metrics & Loss
# =============================================================================

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0; self.cnt = 0
    def update(self, val, n=1): self.sum += val * n; self.cnt += n
    @property
    def avg(self): return self.sum / self.cnt if self.cnt > 0 else 0

def calc_loss(s_logit, a_logits, p_logits, batch, cfg):
    # 1. Script Loss
    l_script = F.binary_cross_entropy_with_logits(s_logit.squeeze(-1), batch['is_correct'])
    
    # 2. Anomaly Loss (Masked)
    tgt_anomaly = 1.0 - batch['pill_correct']
    l_anom_raw = F.binary_cross_entropy_with_logits(a_logits, tgt_anomaly, reduction='none')
    valid = ~batch['pill_mask']
    l_anom = (l_anom_raw * valid).sum() / (valid.sum() + 1e-6)
    
    # 3. Pill Classification Loss (Masked, ignore padding)
    B, N, C = p_logits.shape
    l_cls = F.cross_entropy(p_logits.view(B*N, C), batch['labels'].view(B*N), ignore_index=-100)
    
    total = (cfg.script_loss_weight * l_script + 
             cfg.pill_anomaly_loss_weight * l_anom + 
             cfg.pill_classification_loss_weight * l_cls)
             
    return total, {'loss': total.item(), 'l_script': l_script.item(), 'l_anom': l_anom.item(), 'l_cls': l_cls.item()}

def calc_acc(s_logit, a_logits, p_logits, batch):
    with torch.no_grad():
        # Script Acc
        s_pred = (torch.sigmoid(s_logit.squeeze(-1)) > 0.5).float()
        acc_s = (s_pred == batch['is_correct']).float().mean().item()
        
        # Pill Acc
        valid = ~batch['pill_mask']
        if valid.any():
            p_pred = p_logits.argmax(-1)
            valid_lbl = (batch['labels'] != -100)
            acc_p = (p_pred == batch['labels'])[valid_lbl].float().mean().item()
        else:
            acc_p = 0.0
            
    return {'acc_script': acc_s, 'acc_pill': acc_p}


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, loader, opt, scaler, ddp, epoch, cfg, logger):
    model.train()
    meters = defaultdict(AverageMeter)
    
    if ddp.is_main:
        pbar = tqdm(loader, desc=f"Ep {epoch}", dynamic_ncols=True)
        
    for i, batch in enumerate(loader):
        # Move to GPU
        batch = {k: v.to(ddp.device, non_blocking=True) for k,v in batch.items()}
        
        # Accumulation Logic
        is_sync = (i + 1) % cfg.gradient_accumulation_steps == 0
        
        with ddp_sync_context(model, is_sync):
            with autocast(enabled=cfg.use_amp):
                s_logit, a_logits, p_logits = model(
                    batch['images'], batch['expected_ndcs'], batch['expected_counts'],
                    batch['pill_mask'], batch['expected_mask']
                )
                
                loss, loss_items = calc_loss(s_logit, a_logits, p_logits, batch, cfg)
                loss = loss / cfg.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
        if is_sync:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
        # Logging
        metrics = calc_acc(s_logit, a_logits, p_logits, batch)
        bs = batch['images'].size(0)
        
        for k,v in loss_items.items(): meters[k].update(v, bs)
        for k,v in metrics.items(): meters[k].update(v, bs)
        
        if ddp.is_main:
            pbar.set_postfix({'loss': meters['loss'].avg, 'acc_s': meters['acc_script'].avg})
            
    if ddp.is_main: return {k: v.avg for k,v in meters.items()}
    return None

@torch.no_grad()
def validate(model, loader, ddp, cfg):
    model.eval()
    meters = defaultdict(AverageMeter)
    
    for batch in loader:
        batch = {k: v.to(ddp.device, non_blocking=True) for k,v in batch.items()}
        
        with autocast(enabled=cfg.use_amp):
            s_logit, a_logits, p_logits = model(
                batch['images'], batch['expected_ndcs'], batch['expected_counts'],
                batch['pill_mask'], batch['expected_mask']
            )
            loss, loss_items = calc_loss(s_logit, a_logits, p_logits, batch, cfg)
            metrics = calc_acc(s_logit, a_logits, p_logits, batch)
            
        bs = batch['images'].size(0)
        for k,v in loss_items.items(): meters[k].update(v, bs)
        for k,v in metrics.items(): meters[k].update(v, bs)
        
    # Sync metrics across GPUs
    results = {}
    for k, v in meters.items():
        t = torch.tensor([v.sum, v.cnt], device=ddp.device)
        dist.all_reduce(t)
        results[k] = (t[0] / t[1]).item()
        
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--build-index', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    # 1. Index Building Mode
    if args.build_index:
        DatasetIndexer.build_index(args.data_dir, os.path.join(args.data_dir, "dataset_index.csv"))
        return

    # 2. Setup
    cfg = TrainingConfig(data_dir=args.data_dir, output_dir=args.output_dir, epochs=args.epochs, resume_from=args.resume)
    ddp = DDPManager(cfg)
    ddp.setup()
    logger = setup_logging(ddp.rank, cfg.output_dir)
    
    # A100 Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 3. Data
    idx_path = os.path.join(cfg.data_dir, cfg.index_file)
    if not os.path.exists(idx_path):
        if ddp.is_main: logger.error(f"Index not found at {idx_path}. Run --build-index first.")
        return

    data, cls_map = DatasetIndexer.load_index(idx_path)
    cfg.num_classes = len(cls_map)
    if ddp.is_main: logger.info(f"Loaded {cfg.num_classes} classes.")
    
    train_ds = PrescriptionDataset(cfg.data_dir, data['train'], cls_map, is_training=True, max_pills=cfg.max_pills_per_prescription)
    val_ds = PrescriptionDataset(cfg.data_dir, data['valid'], cls_map, is_training=False, max_pills=cfg.max_pills_per_prescription)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp.is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp.is_distributed else None
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, 
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler, 
                            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # 4. Model
    model = EndToEndModel(cfg).to(ddp.device)
    
    # Apply SyncBatchNorm (CRITICAL for small batches)
    if ddp.is_distributed:
        if ddp.is_main: logger.info("Applying SyncBatchNorm...")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp.local_rank], find_unused_parameters=cfg.find_unused_parameters)
    
    # 5. Optimizer
    # Separate LRs for Backbone vs Verifier
    p_backbone = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    p_verifier = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    
    opt = torch.optim.AdamW([
        {'params': p_backbone, 'lr': cfg.backbone_lr},
        {'params': p_verifier, 'lr': cfg.verifier_lr}
    ], weight_decay=cfg.weight_decay)
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Resume?
    start_ep = 0
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['epoch'] + 1
        if ddp.is_main: logger.info(f"Resumed from epoch {start_ep}")

    # 6. Loop
    for ep in range(start_ep, cfg.epochs):
        if ddp.is_distributed: train_sampler.set_epoch(ep)
        
        # Train
        train_res = train_one_epoch(model, train_loader, opt, scaler, ddp, ep, cfg, logger)
        if ddp.is_main:
            logger.info(f"Ep {ep} Train: Loss={train_res['loss']:.4f} ScriptAcc={train_res['acc_script']:.4f} PillAcc={train_res['acc_pill']:.4f}")
            
        # Validate
        val_res = validate(model, val_loader, ddp, cfg)
        if ddp.is_main:
            logger.info(f"Ep {ep} Val  : Loss={val_res['loss']:.4f} ScriptAcc={val_res['acc_script']:.4f} PillAcc={val_res['acc_pill']:.4f}")
            
            # Save
            if (ep + 1) % cfg.save_every_n_epochs == 0:
                torch.save({
                    'epoch': ep,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'scaler': scaler.state_dict()
                }, os.path.join(cfg.output_dir, f'ckpt_ep{ep}.pt'))
                
    ddp.cleanup()

if __name__ == '__main__':
    main()