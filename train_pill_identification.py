#!/usr/bin/env python3
"""
End-to-End Prescription-Aware Pill Verification Training (FIXED)
=================================================================
Target Hardware: NVIDIA DGX A100 (8x A100 40GB GPUs)

KEY FIXES:
1. SYNCBATCHNORM: Critical for batch_size=2 across 8 GPUs
2. PADDING PROTECTION: Filter padding images BEFORE backbone to prevent
   BatchNorm corruption from zero-filled tensors
3. CURRICULUM LEARNING:
   - Phase 1 (BACKBONE): Train only ResNet classifier until pill_acc > 15%
   - Phase 2 (END_TO_END): Enable verifier losses for script/anomaly detection
4. 640x640 INPUT: Higher resolution for better pill detail recognition

Usage:
    # Build index first
    torchrun --nproc_per_node=8 train_e2e_fixed.py --data-dir /path/to/data --build-index
    
    # Train
    torchrun --nproc_per_node=8 train_e2e_fixed.py --data-dir /path/to/data --epochs 100

    # Background execution
    nohup torchrun --nproc_per_node=8 train_e2e_fixed.py --data-dir /path/to/data > train.log 2>&1 &
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
from dataclasses import dataclass, field
from contextlib import contextmanager

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

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with all fixes applied."""
    
    # Data paths
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Model architecture
    num_classes: int = 1228
    backbone: str = "resnet34"
    pill_embed_dim: int = 512
    verifier_hidden_dim: int = 256
    verifier_num_heads: int = 8
    verifier_num_layers: int = 4
    verifier_dropout: float = 0.1
    
    # Input size (640x640 for better pill detail)
    input_size: int = 640
    
    # Prescription constraints
    max_pills: int = 200
    min_pills: int = 5
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 2  # Per GPU - requires SyncBatchNorm
    grad_accum_steps: int = 4  # Effective batch = 2 * 4 * 8 = 64
    
    # Learning rates
    backbone_lr: float = 5e-5
    verifier_lr: float = 1e-4
    weight_decay: float = 0.01
    
    # LR schedule
    warmup_epochs: int = 3
    min_lr: float = 1e-7
    
    # Loss weights (Phase 2 only)
    script_loss_weight: float = 1.0
    anomaly_loss_weight: float = 2.0  # Boosted for anomaly detection
    cls_loss_weight: float = 1.0
    
    # Curriculum learning
    curriculum_threshold: float = 0.15  # Switch to Phase 2 when val_acc > 15%
    
    # Error injection
    error_rate: float = 0.5
    
    # Hardware optimizations
    use_amp: bool = True
    use_gradient_checkpointing: bool = True  # Required for 640x640
    max_grad_norm: float = 1.0
    
    # DDP settings
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = 30
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 5
    validate_every_n_epochs: int = 1
    log_every_n_steps: int = 10
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None


# =============================================================================
# Logging
# =============================================================================

def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """Setup logging - only rank 0 logs to console and file."""
    logger = logging.getLogger("PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers = []
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
        
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
        
        fh = logging.FileHandler(os.path.join(output_dir, f'train_{datetime.now():%Y%m%d_%H%M%S}.log'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    return logger


# =============================================================================
# DDP Manager
# =============================================================================

class DDPManager:
    """Manages Distributed Data Parallel setup."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1
        self.device = torch.device(f'cuda:{self.local_rank}')
    
    def setup(self) -> None:
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            
            os.environ['NCCL_DEBUG'] = 'WARN'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            
            dist.init_process_group(
                backend=self.config.dist_backend,
                timeout=timedelta(minutes=self.config.dist_timeout_minutes)
            )
            dist.barrier()
            
            if self.rank == 0:
                print(f"DDP initialized: {self.world_size} GPUs")
        else:
            print("Running in single-GPU mode")
    
    def cleanup(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()


# =============================================================================
# Dataset Indexing
# =============================================================================

class DatasetIndexer:
    """Fast CSV-based dataset indexing."""
    
    # Pattern: {ndc}_{prescription_id}_patch_{patch_no}.jpg
    FILENAME_PATTERN = re.compile(r'^(.+)_(.+)_patch_(\d+)\.jpg$', re.IGNORECASE)
    
    @classmethod
    def build_index(cls, data_dir: str, output_file: str, 
                    splits: List[str] = ['train', 'valid']) -> Dict[str, int]:
        print(f"Building index for {data_dir}...")
        stats = defaultdict(int)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'ndc', 'prescription_id', 'patch_no', 'relative_path'])
            
            for split in splits:
                split_dir = Path(data_dir) / split
                if not split_dir.exists():
                    print(f"  Warning: {split_dir} not found")
                    continue
                
                ndc_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
                print(f"  {split}: {len(ndc_dirs)} NDC directories")
                
                for ndc_dir in tqdm(ndc_dirs, desc=f"Indexing {split}"):
                    ndc = ndc_dir.name
                    
                    for img_path in ndc_dir.glob('*.jpg'):
                        match = cls.FILENAME_PATTERN.match(img_path.name)
                        if match:
                            _, rx_id, patch_no = match.groups()
                            rel_path = str(img_path.relative_to(data_dir))
                            writer.writerow([split, ndc, rx_id, int(patch_no), rel_path])
                            stats[f'{split}_images'] += 1
                
                stats[f'{split}_ndcs'] = len(ndc_dirs)
        
        print(f"\nIndex saved: {output_file}")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:,}")
        
        return dict(stats)
    
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


# =============================================================================
# Dataset
# =============================================================================

class PrescriptionDataset(Dataset):
    """Prescription-level dataset with error injection."""
    
    def __init__(
        self,
        data_dir: str,
        split_data: List[dict],
        class_to_idx: Dict[str, int],
        config: TrainingConfig,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.config = config
        self.is_training = is_training
        
        # Transforms for 640x640 input
        self.transform = self._create_transform(is_training, config.input_size)
        
        # Group by prescription
        self.prescriptions = self._group_by_prescription(split_data)
        self.prescription_ids = list(self.prescriptions.keys())
        
        # Build pill library for error injection
        if is_training:
            self.pill_library = self._build_pill_library()
        else:
            self.pill_library = {}
        
        # Track load failures
        self.load_failures = 0
    
    def _create_transform(self, is_training: bool, size: int):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if is_training:
            return transforms.Compose([
                transforms.Resize((size + 32, size + 32)),  # Slight overscan for crop
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize
            ])
    
    def _group_by_prescription(self, data: List[dict]) -> Dict[str, List[dict]]:
        prescriptions = defaultdict(list)
        for item in data:
            prescriptions[item['prescription_id']].append(item)
        
        return {
            rx_id: sorted(pills, key=lambda x: x['patch_no'])
            for rx_id, pills in prescriptions.items()
            if self.config.min_pills <= len(pills) <= self.config.max_pills
        }
    
    def _build_pill_library(self) -> Dict[str, List[str]]:
        library = defaultdict(list)
        for pills in self.prescriptions.values():
            for pill in pills:
                library[pill['ndc']].append(pill['relative_path'])
        return dict(library)
    
    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        try:
            img_path = self.data_dir / rel_path
            img = Image.open(img_path)
            img.load()  # Force load to catch errors
            return img.convert('RGB')
        except Exception as e:
            self.load_failures += 1
            if self.load_failures <= 5:
                print(f"WARNING: Failed to load {rel_path}: {e}")
            return None
    
    def _inject_errors(self, pills: List[dict]) -> Tuple[List[dict], List[int]]:
        """Inject random errors into prescription."""
        n = len(pills)
        modified = [p.copy() for p in pills]
        
        # Determine number of errors
        error_type = random.choices(['single', 'few', 'many'], weights=[0.5, 0.35, 0.15])[0]
        if error_type == 'single':
            n_wrong = 1
        elif error_type == 'few':
            n_wrong = random.randint(2, min(5, max(2, n // 4)))
        else:
            n_wrong = random.randint(max(2, n // 4), max(3, n // 2))
        
        wrong_indices = random.sample(range(n), min(n_wrong, n))
        
        for idx in wrong_indices:
            orig_ndc = modified[idx]['ndc']
            available = [ndc for ndc in self.pill_library.keys() if ndc != orig_ndc]
            
            if available:
                wrong_ndc = random.choice(available)
                modified[idx] = {
                    'relative_path': random.choice(self.pill_library[wrong_ndc]),
                    'ndc': wrong_ndc,
                    'patch_no': modified[idx]['patch_no'],
                    'prescription_id': modified[idx]['prescription_id']
                }
        
        return modified, wrong_indices
    
    def __len__(self):
        return len(self.prescription_ids)
    
    def __getitem__(self, idx):
        rx_id = self.prescription_ids[idx]
        original_pills = self.prescriptions[rx_id]
        
        # Error injection (training only)
        inject_error = self.is_training and random.random() < self.config.error_rate
        
        if inject_error:
            pills, wrong_indices = self._inject_errors(original_pills)
            is_correct = False
        else:
            pills = original_pills
            wrong_indices = []
            is_correct = True
        
        # Load images and labels
        images = []
        labels = []
        valid_flags = []  # Track which images loaded successfully
        
        for pill in pills:
            img = self._load_image(pill['relative_path'])
            
            if img is not None:
                images.append(self.transform(img))
                valid_flags.append(True)
            else:
                # Create placeholder - will be filtered by padding protection
                images.append(torch.zeros(3, self.config.input_size, self.config.input_size))
                valid_flags.append(False)
            
            labels.append(self.class_to_idx[pill['ndc']])
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        valid_flags = torch.tensor(valid_flags, dtype=torch.bool)
        
        # Pill correctness mask (1 = correct, 0 = wrong)
        pill_correct = torch.ones(len(pills), dtype=torch.float)
        for i in wrong_indices:
            pill_correct[i] = 0.0
        
        return {
            'images': images,
            'labels': labels,
            'num_pills': len(pills),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': pill_correct,
            'valid_flags': valid_flags,
            'prescription_id': rx_id
        }


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate with padding tracking."""
    B = len(batch)
    max_pills = max(b['num_pills'] for b in batch)
    C, H, W = batch[0]['images'].shape[1:]
    
    # Initialize tensors
    images = torch.zeros(B, max_pills, C, H, W)
    labels = torch.full((B, max_pills), -100, dtype=torch.long)  # -100 = ignore
    pill_correct = torch.zeros(B, max_pills)
    mask = torch.ones(B, max_pills, dtype=torch.bool)  # True = padding
    
    is_correct = torch.zeros(B)
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        images[i, :n] = b['images']
        labels[i, :n] = b['labels']
        pill_correct[i, :n] = b['pill_correct']
        mask[i, :n] = ~b['valid_flags']  # Invalid images treated as padding
        is_correct[i] = b['is_correct']
    
    return {
        'images': images,
        'labels': labels,
        'mask': mask,
        'pill_correct': pill_correct,
        'is_correct': is_correct
    }


# =============================================================================
# Models
# =============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet backbone with gradient checkpointing.
    Note: BatchNorm will be converted to SyncBatchNorm externally.
    """
    
    def __init__(self, num_classes: int, use_checkpointing: bool = True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.embed_dim = 512
        
        # Split into stages for checkpointing
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # New classifier for our classes
        self.classifier = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Input should be pre-filtered (no padding).
        
        Args:
            x: (N, 3, H, W) - N valid images only
        Returns:
            embeddings: (N, 512)
            logits: (N, num_classes)
        """
        if self.use_checkpointing and self.training:
            x = checkpoint(self.stem, x, use_reentrant=False)
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        x = self.avgpool(x)
        embeddings = x.flatten(1)
        logits = self.classifier(embeddings)
        
        return embeddings, logits


class PrescriptionVerifier(nn.Module):
    """Transformer-based verifier for prescription-level prediction."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        d = config.verifier_hidden_dim
        
        # Project pill embeddings + confidence
        self.projector = nn.Sequential(
            nn.Linear(config.pill_embed_dim + 1, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.verifier_dropout),
            nn.Linear(d, d)
        )
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.verifier_num_heads,
                dim_feedforward=d * 4,
                dropout=config.verifier_dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.verifier_num_layers
        )
        
        # Learnable summary token for script-level prediction
        self.summary_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        
        # Output heads
        self.script_head = nn.Linear(d, 1)
        self.anomaly_head = nn.Linear(d, 1)
    
    def forward(
        self,
        pill_embeddings: torch.Tensor,
        pill_confidences: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pill_embeddings: (B, N, 512)
            pill_confidences: (B, N, 1)
            mask: (B, N) - True for padding
        Returns:
            script_logit: (B, 1)
            anomaly_logits: (B, N)
        """
        B, N, _ = pill_embeddings.shape
        
        # Project
        x = self.projector(torch.cat([pill_embeddings, pill_confidences], dim=-1))
        
        # Add summary token
        summary = self.summary_token.expand(B, -1, -1)
        x = torch.cat([summary, x], dim=1)  # (B, N+1, d)
        
        # Extend mask for summary token (always valid)
        summary_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([summary_mask, mask], dim=1)
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=full_mask)
        
        # Script prediction from summary token
        script_logit = self.script_head(x[:, 0, :])
        
        # Anomaly prediction from pill tokens
        anomaly_logits = self.anomaly_head(x[:, 1:, :]).squeeze(-1)
        
        return script_logit, anomaly_logits


class EndToEndModel(nn.Module):
    """
    Complete model with PADDING PROTECTION.
    
    Padding images are filtered BEFORE the backbone to prevent
    BatchNorm corruption from zero-filled tensors.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.backbone = ResNetBackbone(config.num_classes, config.use_gradient_checkpointing)
        self.verifier = PrescriptionVerifier(config)
        
        self.embed_dim = self.backbone.embed_dim
        self.num_classes = config.num_classes
    
    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with padding protection.
        
        Args:
            images: (B, N, C, H, W)
            mask: (B, N) - True for padding
        Returns:
            script_logit: (B, 1)
            anomaly_logits: (B, N)
            pill_logits: (B, N, num_classes)
        """
        B, N, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        
        # =====================================================================
        # PADDING PROTECTION: Filter out padding before backbone
        # =====================================================================
        
        # 1. Flatten batch
        flat_images = images.view(-1, C, H, W)  # (B*N, C, H, W)
        flat_mask = mask.view(-1)  # (B*N,)
        
        # 2. Get indices of valid (non-padding) images
        valid_indices = torch.nonzero(~flat_mask, as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            # Edge case: all padding (shouldn't happen)
            embeddings = torch.zeros(B, N, self.embed_dim, device=device, dtype=dtype)
            logits = torch.zeros(B, N, self.num_classes, device=device, dtype=dtype)
        else:
            # 3. Extract valid images only
            valid_images = flat_images[valid_indices]  # (N_valid, C, H, W)
            
            # 4. Run backbone on valid images ONLY
            valid_embeddings, valid_logits = self.backbone(valid_images)
            
            # 5. Scatter results back to full tensors
            full_embeddings = torch.zeros(B * N, self.embed_dim, device=device, dtype=valid_embeddings.dtype)
            full_logits = torch.zeros(B * N, self.num_classes, device=device, dtype=valid_logits.dtype)
            
            full_embeddings.index_copy_(0, valid_indices, valid_embeddings)
            full_logits.index_copy_(0, valid_indices, valid_logits)
            
            # 6. Reshape back to (B, N, ...)
            embeddings = full_embeddings.view(B, N, self.embed_dim)
            logits = full_logits.view(B, N, self.num_classes)
        
        # =====================================================================
        # Verifier
        # =====================================================================
        
        # Compute confidence from logits
        confidences = F.softmax(logits, dim=-1).max(dim=-1).values.unsqueeze(-1)
        
        # Run verifier
        script_logit, anomaly_logits = self.verifier(embeddings, confidences, mask)
        
        return script_logit, anomaly_logits, logits


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Tracks running average."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def compute_losses(
    script_logit: torch.Tensor,
    anomaly_logits: torch.Tensor,
    pill_logits: torch.Tensor,
    is_correct: torch.Tensor,
    pill_correct: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    config: TrainingConfig,
    phase: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute losses with curriculum learning support.
    
    Phase 1 (BACKBONE): Only classification loss
    Phase 2 (END_TO_END): All losses
    """
    
    # 1. Classification loss (always active)
    B, N, C = pill_logits.shape
    cls_loss = F.cross_entropy(
        pill_logits.view(B * N, C),
        labels.view(B * N),
        ignore_index=-100
    )
    
    # 2. Script loss
    script_loss = F.binary_cross_entropy_with_logits(
        script_logit.squeeze(-1), is_correct
    )
    
    # 3. Anomaly loss (masked)
    valid = ~mask
    anomaly_target = 1.0 - pill_correct  # 1 = anomaly
    
    if valid.any():
        anomaly_loss_raw = F.binary_cross_entropy_with_logits(
            anomaly_logits, anomaly_target, reduction='none'
        )
        anomaly_loss = (anomaly_loss_raw * valid.float()).sum() / (valid.sum() + 1e-6)
    else:
        anomaly_loss = torch.tensor(0.0, device=script_logit.device)
    
    # Combine based on phase
    if phase == "BACKBONE":
        # Phase 1: Only classification loss
        total_loss = config.cls_loss_weight * cls_loss
    else:
        # Phase 2: All losses
        total_loss = (
            config.cls_loss_weight * cls_loss +
            config.script_loss_weight * script_loss +
            config.anomaly_loss_weight * anomaly_loss
        )
    
    return total_loss, {
        'total': total_loss.item(),
        'cls': cls_loss.item(),
        'script': script_loss.item(),
        'anomaly': anomaly_loss.item() if isinstance(anomaly_loss, torch.Tensor) else anomaly_loss
    }


def compute_metrics(
    script_logit: torch.Tensor,
    anomaly_logits: torch.Tensor,
    pill_logits: torch.Tensor,
    is_correct: torch.Tensor,
    pill_correct: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, float]:
    """Compute accuracy metrics."""
    
    with torch.no_grad():
        # Script accuracy
        script_pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
        script_acc = (script_pred == is_correct).float().mean().item()
        
        # Anomaly accuracy
        valid = ~mask
        if valid.any():
            anomaly_pred = (torch.sigmoid(anomaly_logits) > 0.5).float()
            anomaly_target = 1.0 - pill_correct
            anomaly_acc = ((anomaly_pred == anomaly_target) & valid).float().sum() / valid.float().sum()
            anomaly_acc = anomaly_acc.item()
        else:
            anomaly_acc = 0.0
        
        # Pill classification accuracy
        valid_cls = (labels != -100)
        if valid_cls.any():
            pill_pred = pill_logits.argmax(dim=-1)
            pill_acc = (pill_pred[valid_cls] == labels[valid_cls]).float().mean().item()
            
            # Top-5 accuracy
            _, top5 = pill_logits.view(-1, pill_logits.size(-1)).topk(5, dim=-1)
            in_top5 = (top5 == labels.view(-1).unsqueeze(-1)).any(dim=-1)
            pill_acc_top5 = in_top5[valid_cls.view(-1)].float().mean().item()
        else:
            pill_acc = 0.0
            pill_acc_top5 = 0.0
    
    return {
        'script_acc': script_acc,
        'anomaly_acc': anomaly_acc,
        'pill_acc': pill_acc,
        'pill_acc_top5': pill_acc_top5
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    ddp: DDPManager,
    epoch: int,
    phase: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    
    meters = {
        'loss': AverageMeter(),
        'loss_cls': AverageMeter(),
        'loss_script': AverageMeter(),
        'loss_anomaly': AverageMeter(),
        'script_acc': AverageMeter(),
        'pill_acc': AverageMeter(),
        'pill_acc_top5': AverageMeter()
    }
    
    if ddp.is_main_process():
        pbar = tqdm(total=len(loader), desc=f'Epoch {epoch} [{phase}]', dynamic_ncols=True)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        # Move to device
        images = batch['images'].to(ddp.device, non_blocking=True)
        labels = batch['labels'].to(ddp.device, non_blocking=True)
        mask = batch['mask'].to(ddp.device, non_blocking=True)
        is_correct = batch['is_correct'].to(ddp.device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(ddp.device, non_blocking=True)
        
        B = images.size(0)
        
        # Forward
        with autocast(enabled=config.use_amp):
            script_logit, anomaly_logits, pill_logits = model(images, mask)
            
            loss, loss_dict = compute_losses(
                script_logit, anomaly_logits, pill_logits,
                is_correct, pill_correct, labels, mask,
                config, phase
            )
            loss = loss / config.grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Optimizer step
        if (step + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        metrics = compute_metrics(
            script_logit, anomaly_logits, pill_logits,
            is_correct, pill_correct, labels, mask
        )
        
        # Update meters
        meters['loss'].update(loss_dict['total'], B)
        meters['loss_cls'].update(loss_dict['cls'], B)
        meters['loss_script'].update(loss_dict['script'], B)
        meters['loss_anomaly'].update(loss_dict['anomaly'], B)
        meters['script_acc'].update(metrics['script_acc'], B)
        meters['pill_acc'].update(metrics['pill_acc'], B)
        meters['pill_acc_top5'].update(metrics['pill_acc_top5'], B)
        
        if ddp.is_main_process():
            pbar.set_postfix({
                'loss': f'{meters["loss"].avg:.4f}',
                'pill': f'{meters["pill_acc"].avg:.4f}',
                'top5': f'{meters["pill_acc_top5"].avg:.4f}'
            })
            pbar.update(1)
            
            if (step + 1) % config.log_every_n_steps == 0:
                logger.info(
                    f"Step {step+1}/{len(loader)} | "
                    f"Loss: {meters['loss'].avg:.4f} | "
                    f"Cls: {meters['loss_cls'].avg:.4f} | "
                    f"Pill Acc: {meters['pill_acc'].avg:.4f} | "
                    f"Top5: {meters['pill_acc_top5'].avg:.4f}"
                )
    
    if ddp.is_main_process():
        pbar.close()
    
    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    config: TrainingConfig,
    ddp: DDPManager
) -> Dict[str, float]:
    """Validate model."""
    
    model.eval()
    
    total_correct = 0
    total_count = 0
    
    for batch in loader:
        images = batch['images'].to(ddp.device)
        labels = batch['labels'].to(ddp.device)
        mask = batch['mask'].to(ddp.device)
        
        with autocast(enabled=config.use_amp):
            _, _, pill_logits = model(images, mask)
        
        # Pill accuracy
        valid = (labels != -100)
        if valid.any():
            preds = pill_logits.argmax(dim=-1)
            total_correct += (preds[valid] == labels[valid]).sum().item()
            total_count += valid.sum().item()
    
    # Sync across GPUs
    if ddp.is_distributed:
        correct_tensor = torch.tensor(total_correct, device=ddp.device)
        count_tensor = torch.tensor(total_count, device=ddp.device)
        dist.all_reduce(correct_tensor)
        dist.all_reduce(count_tensor)
        total_correct = correct_tensor.item()
        total_count = count_tensor.item()
    
    pill_acc = total_correct / max(total_count, 1)
    
    return {'pill_acc': pill_acc}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    phase: str,
    val_acc: float,
    output_dir: str
):
    """Save checkpoint."""
    state = {
        'epoch': epoch,
        'phase': phase,
        'val_acc': val_acc,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }
    path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, path)
    
    # Also save as best if applicable
    best_path = os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(best_path):
        torch.save(state, best_path)
    else:
        best = torch.load(best_path, map_location='cpu')
        if val_acc > best.get('val_acc', 0):
            torch.save(state, best_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='End-to-End Pill Verification Training')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--build-index', action='store_true', help='Build index and exit')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=args.resume
    )
    
    # Build index mode
    if args.build_index:
        DatasetIndexer.build_index(
            args.data_dir,
            os.path.join(args.data_dir, config.index_file)
        )
        return
    
    # DDP setup
    ddp = DDPManager(config)
    ddp.setup()
    
    # Logging
    logger = setup_logging(ddp.rank, config.output_dir)
    
    # Seeds
    random.seed(config.seed + ddp.rank)
    np.random.seed(config.seed + ddp.rank)
    torch.manual_seed(config.seed + ddp.rank)
    torch.cuda.manual_seed_all(config.seed + ddp.rank)
    
    # CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    logger.info("=" * 60)
    logger.info("End-to-End Pill Verification Training (FIXED)")
    logger.info("=" * 60)
    logger.info(f"Device: {ddp.device}")
    logger.info(f"World size: {ddp.world_size}")
    logger.info(f"Input size: {config.input_size}x{config.input_size}")
    
    # Load index
    index_path = os.path.join(config.data_dir, config.index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found. Run with --build-index first: {index_path}")
    
    data_by_split, class_to_idx = DatasetIndexer.load_index(index_path)
    config.num_classes = len(class_to_idx)
    
    logger.info(f"Classes: {config.num_classes}")
    logger.info(f"Train samples: {len(data_by_split.get('train', []))}")
    logger.info(f"Valid samples: {len(data_by_split.get('valid', []))}")
    
    # Datasets
    train_dataset = PrescriptionDataset(
        config.data_dir, data_by_split['train'], class_to_idx, config, is_training=True
    )
    val_dataset = PrescriptionDataset(
        config.data_dir, data_by_split['valid'], class_to_idx, config, is_training=False
    )
    
    logger.info(f"Train prescriptions: {len(train_dataset)}")
    logger.info(f"Valid prescriptions: {len(val_dataset)}")
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp.is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp.is_distributed else None
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Model
    model = EndToEndModel(config).to(ddp.device)
    
    # CRITICAL: Convert to SyncBatchNorm for small batch sizes
    if ddp.is_distributed:
        logger.info("Converting BatchNorm to SyncBatchNorm (critical for batch_size=2)")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # DDP wrapper with find_unused_parameters for curriculum learning
        model = DDP(
            model,
            device_ids=[ddp.local_rank],
            find_unused_parameters=True  # Required for Phase 1
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer with separate LRs
    backbone_params = []
    verifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            verifier_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.backbone_lr},
        {'params': verifier_params, 'lr': config.verifier_lr}
    ], weight_decay=config.weight_decay)
    
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training state
    start_epoch = 0
    phase = "BACKBONE"  # Start with backbone-only training
    best_val_acc = 0.0
    
    # Resume if specified
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=ddp.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        phase = ckpt.get('phase', 'BACKBONE')
        best_val_acc = ckpt.get('val_acc', 0.0)
    
    # Effective batch size
    eff_batch = config.batch_size * config.grad_accum_steps * ddp.world_size
    logger.info(f"Effective batch size: {config.batch_size} x {config.grad_accum_steps} x {ddp.world_size} = {eff_batch}")
    logger.info(f"Starting phase: {phase}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.epochs):
        if ddp.is_distributed:
            train_sampler.set_epoch(epoch)
        
        logger.info(f"\nEpoch {epoch} [{phase}]")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            config, ddp, epoch, phase, logger
        )
        
        logger.info(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Cls: {train_metrics['loss_cls']:.4f} | "
            f"Pill Acc: {train_metrics['pill_acc']:.4f} | "
            f"Top5: {train_metrics['pill_acc_top5']:.4f}"
        )
        
        # Validate
        val_metrics = validate(model, val_loader, config, ddp)
        val_acc = val_metrics['pill_acc']
        
        logger.info(f"Val   | Pill Acc: {val_acc:.4f}")
        
        # Save checkpoint
        if ddp.is_main_process():
            if (epoch + 1) % config.save_every_n_epochs == 0 or val_acc > best_val_acc:
                save_checkpoint(
                    model, optimizer, scaler, epoch, phase, val_acc, config.output_dir
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Curriculum transition: BACKBONE -> END_TO_END
        if phase == "BACKBONE":
            # Broadcast decision from rank 0
            acc_tensor = torch.tensor(val_acc, device=ddp.device)
            if ddp.is_distributed:
                dist.broadcast(acc_tensor, src=0)
            
            if acc_tensor.item() > config.curriculum_threshold:
                phase = "END_TO_END"
                logger.info(f">>> SWITCHING TO END-TO-END TRAINING (Val Acc {acc_tensor.item():.4f} > {config.curriculum_threshold}) <<<")
        
        # Cleanup
        ddp.barrier()
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    ddp.cleanup()


if __name__ == '__main__':
    main()