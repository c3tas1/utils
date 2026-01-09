#!/usr/bin/env python3
"""
Prescription-Aware Pill Verification Training Script
=====================================================
Designed for DGX A100 (8x A100 40GB GPUs) with robust DDP handling.

Features:
- Fast CSV-based indexing for 100M+ images
- Robust DDP with comprehensive error handling for A100s
- Mixed precision training (AMP) with gradient scaling
- Gradient checkpointing for memory efficiency
- Comprehensive logging with running metrics
- Validation collage generation for intermediate results
- Automatic checkpoint saving and resumption

Usage:
    # Build index first (run once)
    python train_prescription_verifier.py --build-index --data-dir /path/to/data
    
    # Train
    torchrun --nproc_per_node=8 train_prescription_verifier.py \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --epochs 100 \
        --batch-size 4

Author: Claude (Anthropic)
"""

import os
import sys
import re
import csv
import json
import math
import random
import signal
import socket
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for A100."""
    
    # Data paths
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Model architecture
    num_classes: int = 1000  # Number of NDCs
    pill_embed_dim: int = 512  # ResNet34 embedding dimension
    max_pills_per_prescription: int = 200
    min_pills_per_prescription: int = 5
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 4  # Per GPU - prescriptions, not pills
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    script_loss_weight: float = 1.0
    pill_loss_weight: float = 0.5
    
    # Error injection
    error_rate: float = 0.5  # Fraction of samples with synthetic errors
    
    # A100-specific optimizations
    use_amp: bool = True  # Mixed precision
    use_gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # DDP settings
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = 30
    find_unused_parameters: bool = False
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    collage_every_n_epochs: int = 1
    num_collage_samples: int = 16
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """Setup logging - only rank 0 logs to file and console."""
    logger = logging.getLogger("PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DDP Utilities with Robust Error Handling
# =============================================================================

class DDPManager:
    """Manages Distributed Data Parallel setup with robust error handling for A100."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        self.device = torch.device('cuda:0')
        
    def setup(self) -> None:
        """Initialize DDP with comprehensive error handling."""
        
        # Check if we're in a distributed environment
        if 'RANK' not in os.environ:
            print("Running in single-GPU mode")
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            return
        
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.is_distributed = True
        
        # Set device before any CUDA operations
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # A100-specific NCCL optimizations
        os.environ['NCCL_DEBUG'] = 'WARN'  # Set to 'INFO' for debugging
        os.environ['NCCL_IB_DISABLE'] = '0'  # Enable InfiniBand if available
        os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P for NVLink
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Better error handling
        
        # For A100 with NVSwitch, optimize for high bandwidth
        os.environ['NCCL_ALGO'] = 'Ring'  # Ring algorithm works well with NVLink
        os.environ['NCCL_PROTO'] = 'Simple'  # Simple protocol for NVSwitch
        
        # Initialize process group with timeout
        timeout = timedelta(minutes=self.config.dist_timeout_minutes)
        
        try:
            dist.init_process_group(
                backend=self.config.dist_backend,
                timeout=timeout,
                rank=self.rank,
                world_size=self.world_size
            )
        except Exception as e:
            print(f"[Rank {self.rank}] Failed to initialize process group: {e}")
            raise
        
        # Barrier to ensure all processes are ready
        dist.barrier()
        
        if self.rank == 0:
            print(f"DDP initialized: {self.world_size} GPUs")
            print(f"  Backend: {self.config.dist_backend}")
            print(f"  Timeout: {self.config.dist_timeout_minutes} minutes")
    
    def cleanup(self) -> None:
        """Clean up DDP resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce a tensor across all processes."""
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src to all processes."""
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor


@contextmanager
def ddp_sync_context(model: nn.Module, sync: bool = True):
    """
    Context manager for controlling gradient synchronization in DDP.
    Use sync=False for gradient accumulation steps.
    """
    if isinstance(model, DDP):
        with model.no_sync() if not sync else contextmanager(lambda: (yield))():
            yield
    else:
        yield


# =============================================================================
# Fast Dataset Indexing
# =============================================================================

class DatasetIndexer:
    """
    Fast CSV-based indexing for large-scale pill image datasets.
    
    File naming convention: {ndc}_{prescription_id}_{patch_no}.jpg
    Directory structure: {split}/{ndc}/{ndc}_{prescription_id}_{patch_no}.jpg
    """
    
    FILENAME_PATTERN = re.compile(r'^(.+)_(.+)_(\d+)\.jpg$', re.IGNORECASE)
    
    @classmethod
    def build_index(cls, data_dir: str, output_file: str, splits: List[str] = ['train', 'valid']) -> Dict[str, int]:
        """
        Build CSV index for all images in the dataset.
        Returns statistics about the indexed data.
        """
        print(f"Building index for {data_dir}...")
        
        stats = defaultdict(int)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'ndc', 'prescription_id', 'patch_no', 'relative_path'])
            
            for split in splits:
                split_dir = Path(data_dir) / split
                if not split_dir.exists():
                    print(f"  Warning: {split_dir} does not exist, skipping")
                    continue
                
                ndc_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
                print(f"  Processing {split}: {len(ndc_dirs)} NDC directories")
                
                for ndc_dir in tqdm(ndc_dirs, desc=f"Indexing {split}"):
                    ndc = ndc_dir.name
                    
                    for img_path in ndc_dir.glob('*.jpg'):
                        match = cls.FILENAME_PATTERN.match(img_path.name)
                        if match:
                            file_ndc, prescription_id, patch_no = match.groups()
                            relative_path = str(img_path.relative_to(data_dir))
                            
                            writer.writerow([split, ndc, prescription_id, int(patch_no), relative_path])
                            
                            stats[f'{split}_images'] += 1
                            stats[f'{split}_ndcs'] += 0  # Will count unique later
                
                # Count unique NDCs and prescriptions
                stats[f'{split}_ndcs'] = len(ndc_dirs)
        
        # Count unique prescriptions from the CSV
        prescriptions_per_split = defaultdict(set)
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prescriptions_per_split[row['split']].add(row['prescription_id'])
        
        for split in splits:
            stats[f'{split}_prescriptions'] = len(prescriptions_per_split[split])
        
        print(f"\nIndex built: {output_file}")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value:,}")
        
        return dict(stats)
    
    @classmethod
    def load_index(cls, index_file: str) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
        """
        Load pre-built CSV index.
        Returns (data_by_split, class_to_idx mapping)
        """
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
        
        # Create class to index mapping (sorted for consistency)
        class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
        
        return dict(data_by_split), class_to_idx


# =============================================================================
# Dataset Classes
# =============================================================================

class PrescriptionDataset(Dataset):
    """
    Prescription-level dataset for script verification training.
    Uses pre-built CSV index for fast loading.
    """
    
    def __init__(
        self,
        data_dir: str,
        split_data: List[dict],
        class_to_idx: Dict[str, int],
        transform=None,
        max_pills: int = 200,
        min_pills: int = 5,
        error_rate: float = 0.0,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)
        self.transform = transform or self._default_transform(is_training)
        self.max_pills = max_pills
        self.min_pills = min_pills
        self.error_rate = error_rate
        self.is_training = is_training
        
        # Group by prescription
        self.prescriptions = self._group_by_prescription(split_data)
        self.prescription_ids = list(self.prescriptions.keys())
        
        # Build pill library for error injection
        if error_rate > 0:
            self.pill_library = self._build_pill_library()
        else:
            self.pill_library = {}
    
    def _group_by_prescription(self, data: List[dict]) -> Dict[str, List[dict]]:
        """Group pills by prescription ID."""
        prescriptions = defaultdict(list)
        
        for item in data:
            prescriptions[item['prescription_id']].append(item)
        
        # Filter by pill count and sort pills within each prescription
        filtered = {}
        for rx_id, pills in prescriptions.items():
            if self.min_pills <= len(pills) <= self.max_pills:
                filtered[rx_id] = sorted(pills, key=lambda x: x['patch_no'])
        
        return filtered
    
    def _build_pill_library(self) -> Dict[str, List[str]]:
        """Build library of pill paths per NDC for error injection."""
        library = defaultdict(list)
        for rx_id, pills in self.prescriptions.items():
            for pill in pills:
                library[pill['ndc']].append(pill['relative_path'])
        return dict(library)
    
    def _default_transform(self, is_training: bool):
        if is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_random_wrong_pill(self, correct_ndc: str) -> Tuple[str, str, int]:
        """Get a random pill from a different NDC."""
        available_ndcs = [ndc for ndc in self.pill_library.keys() if ndc != correct_ndc]
        if not available_ndcs:
            # Fallback: return same NDC if no alternatives
            return random.choice(self.pill_library[correct_ndc]), correct_ndc, self.class_to_idx[correct_ndc]
        
        wrong_ndc = random.choice(available_ndcs)
        wrong_path = random.choice(self.pill_library[wrong_ndc])
        wrong_class_idx = self.class_to_idx[wrong_ndc]
        return wrong_path, wrong_ndc, wrong_class_idx
    
    def _inject_errors(self, pills: List[dict]) -> Tuple[List[dict], List[int], str]:
        """Inject errors into a prescription."""
        error_type = random.choices(
            ['single', 'few', 'many', 'all'],
            weights=[0.5, 0.3, 0.15, 0.05]
        )[0]
        
        n_pills = len(pills)
        modified_pills = [p.copy() for p in pills]
        
        if error_type == 'single':
            n_wrong = 1
        elif error_type == 'few':
            n_wrong = random.randint(2, min(5, max(2, n_pills // 4)))
        elif error_type == 'many':
            n_wrong = random.randint(max(2, n_pills // 4), max(3, n_pills // 2))
        else:  # all
            n_wrong = n_pills
        
        wrong_indices = random.sample(range(n_pills), min(n_wrong, n_pills))
        
        for idx in wrong_indices:
            original_ndc = modified_pills[idx]['ndc']
            wrong_path, wrong_ndc, wrong_class_idx = self._get_random_wrong_pill(original_ndc)
            
            modified_pills[idx] = {
                'relative_path': wrong_path,
                'ndc': wrong_ndc,
                'patch_no': modified_pills[idx]['patch_no'],
                'prescription_id': modified_pills[idx]['prescription_id'],
                'is_wrong': True,
                'original_ndc': original_ndc
            }
        
        return modified_pills, wrong_indices, error_type
    
    def get_prescription_composition(self, prescription_id: str) -> Dict[str, int]:
        """Get expected NDC composition of a prescription."""
        pills = self.prescriptions[prescription_id]
        composition = defaultdict(int)
        for pill in pills:
            composition[pill['ndc']] += 1
        return dict(composition)
    
    def __len__(self):
        return len(self.prescription_ids)
    
    def __getitem__(self, idx):
        prescription_id = self.prescription_ids[idx]
        original_pills = self.prescriptions[prescription_id]
        
        # Decide whether to inject errors
        inject_error = self.is_training and random.random() < self.error_rate
        
        if inject_error:
            pills, wrong_indices, error_type = self._inject_errors(original_pills)
            is_correct = False
        else:
            pills = original_pills
            wrong_indices = []
            error_type = 'none'
            is_correct = True
        
        # Load pill images
        images = []
        actual_labels = []
        paths = []
        
        for pill in pills:
            img_path = self.data_dir / pill['relative_path']
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                actual_labels.append(self.class_to_idx[pill['ndc']])
                paths.append(pill['relative_path'])
            except Exception as e:
                # Skip corrupted images - use a black placeholder
                if self.transform:
                    img = torch.zeros(3, 224, 224)
                else:
                    img = torch.zeros(3, 224, 224)
                images.append(img)
                actual_labels.append(self.class_to_idx.get(pill['ndc'], 0))
                paths.append(pill['relative_path'])
        
        images = torch.stack(images)
        actual_labels = torch.tensor(actual_labels, dtype=torch.long)
        
        # Expected composition
        composition = self.get_prescription_composition(prescription_id)
        expected_ndcs = [self.class_to_idx[ndc] for ndc in composition.keys()]
        expected_counts = list(composition.values())
        
        # Per-pill correctness
        pill_correct = torch.ones(len(pills), dtype=torch.float)
        for idx in wrong_indices:
            pill_correct[idx] = 0.0
        
        return {
            'images': images,
            'actual_labels': actual_labels,
            'prescription_id': prescription_id,
            'num_pills': len(pills),
            'expected_ndcs': torch.tensor(expected_ndcs, dtype=torch.long),
            'expected_counts': torch.tensor(expected_counts, dtype=torch.float),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': pill_correct,
            'wrong_indices': wrong_indices,
            'error_type': error_type,
            'paths': paths
        }


def prescription_collate_fn(batch: List[dict]) -> dict:
    """Custom collate function for variable-length prescriptions."""
    max_pills = max(item['num_pills'] for item in batch)
    max_expected = max(len(item['expected_ndcs']) for item in batch)
    batch_size = len(batch)
    
    # Get dimensions
    C, H, W = batch[0]['images'].shape[1:]
    
    # Initialize padded tensors
    images_padded = torch.zeros(batch_size, max_pills, C, H, W)
    actual_labels_padded = torch.full((batch_size, max_pills), -1, dtype=torch.long)
    pill_correct_padded = torch.ones(batch_size, max_pills)
    pill_mask = torch.ones(batch_size, max_pills, dtype=torch.bool)  # True = padded
    
    expected_ndcs_padded = torch.zeros(batch_size, max_expected, dtype=torch.long)
    expected_counts_padded = torch.zeros(batch_size, max_expected, dtype=torch.float)
    expected_mask = torch.ones(batch_size, max_expected, dtype=torch.bool)
    
    is_correct = torch.zeros(batch_size)
    num_pills = torch.zeros(batch_size, dtype=torch.long)
    prescription_ids = []
    all_paths = []
    all_wrong_indices = []
    
    for i, item in enumerate(batch):
        n = item['num_pills']
        n_expected = len(item['expected_ndcs'])
        
        images_padded[i, :n] = item['images']
        actual_labels_padded[i, :n] = item['actual_labels']
        pill_correct_padded[i, :n] = item['pill_correct']
        pill_mask[i, :n] = False
        
        expected_ndcs_padded[i, :n_expected] = item['expected_ndcs']
        expected_counts_padded[i, :n_expected] = item['expected_counts']
        expected_mask[i, :n_expected] = False
        
        is_correct[i] = item['is_correct']
        num_pills[i] = n
        prescription_ids.append(item['prescription_id'])
        all_paths.append(item['paths'])
        all_wrong_indices.append(item['wrong_indices'])
    
    return {
        'images': images_padded,
        'actual_labels': actual_labels_padded,
        'pill_correct': pill_correct_padded,
        'pill_mask': pill_mask,
        'expected_ndcs': expected_ndcs_padded,
        'expected_counts': expected_counts_padded,
        'expected_mask': expected_mask,
        'is_correct': is_correct,
        'num_pills': num_pills,
        'prescription_ids': prescription_ids,
        'paths': all_paths,
        'wrong_indices': all_wrong_indices
    }


# =============================================================================
# Model Architecture
# =============================================================================

class PrescriptionAwareVerifier(nn.Module):
    """
    Prescription-aware verification model using Set Transformer architecture.
    Takes pill embeddings and prescription context to predict script correctness.
    """
    
    def __init__(
        self,
        pill_embed_dim: int = 512,
        ndc_vocab_size: int = 1000,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # NDC embedding for prescription encoding
        self.ndc_embedding = nn.Embedding(ndc_vocab_size, hidden_dim)
        
        # Prescription encoder
        self.prescription_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            ),
            num_layers=2
        )
        
        # Project pill features to hidden dimension
        self.pill_projector = nn.Sequential(
            nn.Linear(pill_embed_dim + 1, hidden_dim),  # +1 for confidence
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention: pills attend to prescription
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        self.cross_attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(2)
        ])
        
        # Self-attention for pill interactions
        self.pill_encoder = nn.TransformerEncoder(
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
        
        # Learnable summary token for set aggregation
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Final attention pooling
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # Classification heads
        self.script_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.pill_anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _cross_attention_block(self, x, context, context_mask, layer_idx):
        """Cross-attention with residual connection."""
        attn_out, _ = self.cross_attention_layers[layer_idx](
            query=x,
            key=context,
            value=context,
            key_padding_mask=context_mask
        )
        return self.cross_attention_norms[layer_idx](x + attn_out)
    
    def forward(
        self,
        pill_embeddings: torch.Tensor,
        pill_confidences: torch.Tensor,
        prescription_ndcs: torch.Tensor,
        prescription_counts: torch.Tensor,
        pill_mask: Optional[torch.Tensor] = None,
        prescription_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pill_embeddings: (B, N_pills, pill_embed_dim)
            pill_confidences: (B, N_pills, 1)
            prescription_ndcs: (B, N_rx)
            prescription_counts: (B, N_rx, 1)
            pill_mask: (B, N_pills) - True for padded positions
            prescription_mask: (B, N_rx) - True for padded positions
            
        Returns:
            script_logits: (B, 1)
            pill_anomaly: (B, N_pills)
        """
        B = pill_embeddings.size(0)
        
        # Encode prescription
        rx_embed = self.ndc_embedding(prescription_ndcs)  # (B, N_rx, hidden_dim)
        
        # Add count information
        count_scale = prescription_counts / (prescription_counts.sum(dim=1, keepdim=True) + 1e-8)
        rx_embed = rx_embed + count_scale * 0.1
        
        # Encode prescription context
        if self.use_gradient_checkpointing and self.training:
            rx_encoded = checkpoint(
                self.prescription_encoder,
                rx_embed,
                use_reentrant=False
            )
        else:
            rx_encoded = self.prescription_encoder(
                rx_embed,
                src_key_padding_mask=prescription_mask
            )
        
        # Project pill features
        pill_features = torch.cat([pill_embeddings, pill_confidences], dim=-1)
        pill_projected = self.pill_projector(pill_features)
        
        # Cross-attention: pills attend to prescription
        for i in range(len(self.cross_attention_layers)):
            pill_projected = self._cross_attention_block(
                pill_projected, rx_encoded, prescription_mask, i
            )
        
        # Self-attention among pills
        if self.use_gradient_checkpointing and self.training:
            pill_encoded = checkpoint(
                self.pill_encoder,
                pill_projected,
                use_reentrant=False
            )
        else:
            pill_encoded = self.pill_encoder(
                pill_projected,
                src_key_padding_mask=pill_mask
            )
        
        # Per-pill anomaly scores
        pill_anomaly = self.pill_anomaly_head(pill_encoded).squeeze(-1)  # (B, N_pills)
        
        # Set-level aggregation
        summary = self.summary_token.expand(B, -1, -1)
        set_summary, _ = self.pool_attention(
            query=summary,
            key=pill_encoded,
            value=pill_encoded,
            key_padding_mask=pill_mask
        )
        set_summary = self.pool_norm(summary + set_summary)
        
        # Script-level classification
        script_logits = self.script_classifier(set_summary.squeeze(1))
        
        return script_logits, pill_anomaly


class EmbeddingExtractor(nn.Module):
    """
    Wrapper for pretrained ResNet to extract pill embeddings.
    Can optionally include a classification head.
    """
    
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        num_classes: int = 1000,
        freeze: bool = True
    ):
        super().__init__()
        
        # Load pretrained ResNet34
        if pretrained_path and os.path.exists(pretrained_path):
            self.resnet = models.resnet34(weights=None)
            # Load pretrained weights
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            # Handle potential key mismatches
            self.resnet.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            print("Using ImageNet pretrained weights")
        
        # Get embedding dimension
        self.embed_dim = self.resnet.fc.in_features  # 512 for ResNet34
        
        # Replace FC layer with identity for embeddings
        self.resnet.fc = nn.Identity()
        
        # Optional classifier
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings and optionally classify.
        
        Args:
            x: (B, C, H, W) or (B, N, C, H, W)
            
        Returns:
            embeddings: (..., embed_dim)
            logits: (..., num_classes)
        """
        original_shape = x.shape
        
        # Handle batched pill images
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
        
        embeddings = self.resnet(x)
        logits = self.classifier(embeddings)
        
        # Reshape back
        if len(original_shape) == 5:
            embeddings = embeddings.view(B, N, -1)
            logits = logits.view(B, N, -1)
        
        return embeddings, logits


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        if n <= 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    @property
    def is_empty(self) -> bool:
        return self.count == 0


class LRScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        base_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def compute_metrics(
    script_logits: torch.Tensor,
    pill_anomaly: torch.Tensor,
    script_labels: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor
) -> Dict[str, float]:
    """Compute training/validation metrics."""
    
    with torch.no_grad():
        # Script-level accuracy
        script_probs = torch.sigmoid(script_logits.squeeze(-1))
        script_preds = (script_probs > 0.5).float()
        script_acc = (script_preds == script_labels).float().mean().item()
        
        # Pill-level accuracy (masked)
        pill_probs = torch.sigmoid(pill_anomaly)
        pill_preds = (pill_probs > 0.5).float()
        pill_labels_binary = 1 - pill_labels  # Convert correct=1 to anomaly=0
        
        valid_mask = ~pill_mask
        if valid_mask.any():
            pill_acc = (
                (pill_preds == pill_labels_binary)[valid_mask].float().mean().item()
            )
        else:
            pill_acc = 0.0
        
        # Script-level metrics by error type
        correct_scripts = script_labels == 1
        incorrect_scripts = script_labels == 0
        
        correct_script_acc = 0.0
        incorrect_script_acc = 0.0
        
        if correct_scripts.any():
            correct_script_acc = (
                script_preds[correct_scripts] == script_labels[correct_scripts]
            ).float().mean().item()
        
        if incorrect_scripts.any():
            incorrect_script_acc = (
                script_preds[incorrect_scripts] == script_labels[incorrect_scripts]
            ).float().mean().item()
    
    return {
        'script_acc': script_acc,
        'pill_acc': pill_acc,
        'correct_script_acc': correct_script_acc,
        'incorrect_script_acc': incorrect_script_acc
    }


def compute_loss(
    script_logits: torch.Tensor,
    pill_anomaly: torch.Tensor,
    script_labels: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor,
    script_weight: float = 1.0,
    pill_weight: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss."""
    
    # Script-level loss
    script_loss = F.binary_cross_entropy_with_logits(
        script_logits.squeeze(-1), script_labels
    )
    
    # Pill-level loss (masked)
    pill_labels_binary = 1 - pill_labels  # Convert correct=1 to anomaly=0
    pill_loss_raw = F.binary_cross_entropy_with_logits(
        pill_anomaly, pill_labels_binary, reduction='none'
    )
    
    valid_mask = ~pill_mask
    if valid_mask.any():
        pill_loss = (pill_loss_raw * valid_mask.float()).sum() / valid_mask.sum()
    else:
        pill_loss = torch.tensor(0.0, device=script_loss.device)
    
    # Combined loss
    total_loss = script_weight * script_loss + pill_weight * pill_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'script': script_loss.item(),
        'pill': pill_loss.item()
    }


# =============================================================================
# Visualization
# =============================================================================

def create_validation_collage(
    model: nn.Module,
    embedding_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_path: str,
    num_samples: int = 16,
    epoch: int = 0
):
    """Create a visualization collage of validation predictions."""
    
    model.eval()
    embedding_model.eval()
    
    samples = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(samples) >= num_samples:
                break
            
            images = batch['images'].to(device)
            pill_mask = batch['pill_mask'].to(device)
            expected_ndcs = batch['expected_ndcs'].to(device)
            expected_counts = batch['expected_counts'].to(device)
            expected_mask = batch['expected_mask'].to(device)
            is_correct = batch['is_correct'].to(device)
            pill_correct = batch['pill_correct'].to(device)
            num_pills = batch['num_pills']
            prescription_ids = batch['prescription_ids']
            wrong_indices = batch['wrong_indices']
            
            B, max_pills, C, H, W = images.shape
            
            # Get embeddings
            images_flat = images.view(B * max_pills, C, H, W)
            embeddings_flat, logits_flat = embedding_model(images_flat)
            pill_embeddings = embeddings_flat.view(B, max_pills, -1)
            pill_logits = logits_flat.view(B, max_pills, -1)
            
            # Get confidences
            pill_probs = F.softmax(pill_logits, dim=-1)
            pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
            
            # Forward through verifier
            script_logits, pill_anomaly = model(
                pill_embeddings=pill_embeddings,
                pill_confidences=pill_confidences,
                prescription_ndcs=expected_ndcs,
                prescription_counts=expected_counts.unsqueeze(-1),
                pill_mask=pill_mask,
                prescription_mask=expected_mask
            )
            
            script_probs = torch.sigmoid(script_logits.squeeze(-1))
            pill_anomaly_probs = torch.sigmoid(pill_anomaly)
            
            for i in range(B):
                if len(samples) >= num_samples:
                    break
                
                n = num_pills[i].item()
                
                # Denormalize images for visualization
                sample_images = images[i, :n].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                sample_images = sample_images * std + mean
                sample_images = sample_images.clamp(0, 1)
                
                samples.append({
                    'images': sample_images,
                    'num_pills': n,
                    'prescription_id': prescription_ids[i],
                    'is_correct_gt': is_correct[i].item(),
                    'script_prob': script_probs[i].item(),
                    'pill_anomaly_probs': pill_anomaly_probs[i, :n].cpu().numpy(),
                    'pill_correct_gt': pill_correct[i, :n].cpu().numpy(),
                    'wrong_indices': wrong_indices[i]
                })
    
    # Create collage
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Create mini-grid of pills (max 4x4)
        n_pills = min(sample['num_pills'], 16)
        grid_size = int(np.ceil(np.sqrt(n_pills)))
        
        pill_grid = np.ones((grid_size * 64, grid_size * 64, 3))
        
        for p_idx in range(n_pills):
            r, c = p_idx // grid_size, p_idx % grid_size
            pill_img = sample['images'][p_idx].permute(1, 2, 0).numpy()
            pill_img = np.array(Image.fromarray(
                (pill_img * 255).astype(np.uint8)
            ).resize((60, 60)))
            
            # Add border based on prediction
            anomaly_prob = sample['pill_anomaly_probs'][p_idx]
            is_wrong_gt = p_idx in sample['wrong_indices']
            
            if is_wrong_gt:
                border_color = [1, 0, 0]  # Red for wrong
            else:
                border_color = [0, 1, 0]  # Green for correct
            
            # Draw border
            pill_padded = np.ones((64, 64, 3)) * (1 - anomaly_prob)
            pill_padded[2:62, 2:62] = pill_img / 255.0
            
            if is_wrong_gt:
                pill_padded[:2, :] = border_color
                pill_padded[-2:, :] = border_color
                pill_padded[:, :2] = border_color
                pill_padded[:, -2:] = border_color
            
            pill_grid[r*64:(r+1)*64, c*64:(c+1)*64] = pill_padded
        
        ax.imshow(pill_grid)
        
        # Title with prediction info
        gt_label = "CORRECT" if sample['is_correct_gt'] else "WRONG"
        pred_label = "CORRECT" if sample['script_prob'] > 0.5 else "WRONG"
        color = 'green' if (sample['script_prob'] > 0.5) == sample['is_correct_gt'] else 'red'
        
        ax.set_title(
            f"Rx: {sample['prescription_id'][:10]}...\n"
            f"GT: {gt_label} | Pred: {pred_label} ({sample['script_prob']:.2f})\n"
            f"Pills: {sample['num_pills']} | Wrong: {len(sample['wrong_indices'])}",
            fontsize=10,
            color=color
        )
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(samples), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Validation Results - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    embedding_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: TrainingConfig,
    output_dir: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'embedding_model_state_dict': embedding_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
        'config': config.__dict__
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
    
    # Clean up old checkpoints
    checkpoints = sorted(
        [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    while len(checkpoints) > config.keep_last_n_checkpoints:
        old_checkpoint = checkpoints.pop(0)
        os.remove(os.path.join(output_dir, old_checkpoint))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    embedding_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    embedding_model.load_state_dict(checkpoint['embedding_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_metric']


# =============================================================================
# Main Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    embedding_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    ddp_manager: DDPManager,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    embedding_model.eval()
    
    # Metrics
    loss_meter = AverageMeter('Loss')
    script_loss_meter = AverageMeter('Script Loss')
    pill_loss_meter = AverageMeter('Pill Loss')
    script_acc_meter = AverageMeter('Script Acc')
    pill_acc_meter = AverageMeter('Pill Acc')
    
    # Progress bar (only on rank 0)
    if ddp_manager.is_main_process():
        pbar = tqdm(
            total=len(train_loader),
            desc=f'Epoch {epoch}',
            dynamic_ncols=True
        )
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Move to device
        images = batch['images'].to(ddp_manager.device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(ddp_manager.device, non_blocking=True)
        expected_ndcs = batch['expected_ndcs'].to(ddp_manager.device, non_blocking=True)
        expected_counts = batch['expected_counts'].to(ddp_manager.device, non_blocking=True)
        expected_mask = batch['expected_mask'].to(ddp_manager.device, non_blocking=True)
        is_correct = batch['is_correct'].to(ddp_manager.device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(ddp_manager.device, non_blocking=True)
        
        B, max_pills, C, H, W = images.shape
        
        # Determine if we should sync gradients
        sync_gradients = (step + 1) % config.gradient_accumulation_steps == 0
        
        with ddp_sync_context(model, sync=sync_gradients):
            with autocast(enabled=config.use_amp):
                # Extract pill embeddings (frozen)
                with torch.no_grad():
                    images_flat = images.view(B * max_pills, C, H, W)
                    embeddings_flat, logits_flat = embedding_model(images_flat)
                    pill_embeddings = embeddings_flat.view(B, max_pills, -1)
                    pill_logits = logits_flat.view(B, max_pills, -1)
                    
                    # Compute confidences
                    pill_probs = F.softmax(pill_logits, dim=-1)
                    pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
                
                # Forward through verifier
                script_logits, pill_anomaly = model(
                    pill_embeddings=pill_embeddings,
                    pill_confidences=pill_confidences,
                    prescription_ndcs=expected_ndcs,
                    prescription_counts=expected_counts.unsqueeze(-1),
                    pill_mask=pill_mask,
                    prescription_mask=expected_mask
                )
                
                # Compute loss
                loss, loss_components = compute_loss(
                    script_logits, pill_anomaly,
                    is_correct, pill_correct, pill_mask,
                    config.script_loss_weight, config.pill_loss_weight
                )
                
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        
        # Optimizer step (only when accumulation complete)
        if sync_gradients:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Compute metrics
        metrics = compute_metrics(
            script_logits, pill_anomaly,
            is_correct, pill_correct, pill_mask
        )
        
        # Update meters
        loss_meter.update(loss_components['total'], B)
        script_loss_meter.update(loss_components['script'], B)
        pill_loss_meter.update(loss_components['pill'], B)
        script_acc_meter.update(metrics['script_acc'], B)
        pill_acc_meter.update(metrics['pill_acc'], B)
        
        # Update progress bar
        if ddp_manager.is_main_process():
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'script_loss': f'{script_loss_meter.avg:.4f}',
                'pill_loss': f'{pill_loss_meter.avg:.4f}',
                'train_acc': f'{script_acc_meter.avg:.4f}',
                'script_acc': f'{script_acc_meter.avg:.4f}',
                'pill_acc': f'{pill_acc_meter.avg:.4f}'
            })
            pbar.update(1)
    
    if ddp_manager.is_main_process():
        pbar.close()
    
    # Synchronize metrics across GPUs
    if ddp_manager.is_distributed:
        metrics_tensor = torch.tensor([
            loss_meter.sum, loss_meter.count,
            script_acc_meter.sum, script_acc_meter.count,
            pill_acc_meter.sum, pill_acc_meter.count
        ], device=ddp_manager.device, dtype=torch.float64)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Safe division with epsilon to prevent division by zero
        eps = 1e-8
        loss_count = max(metrics_tensor[1].item(), eps)
        script_count = max(metrics_tensor[3].item(), eps)
        pill_count = max(metrics_tensor[5].item(), eps)
        
        total_loss = metrics_tensor[0].item() / loss_count
        total_script_acc = metrics_tensor[2].item() / script_count
        total_pill_acc = metrics_tensor[4].item() / pill_count
    else:
        total_loss = loss_meter.avg if loss_meter.count > 0 else 0.0
        total_script_acc = script_acc_meter.avg if script_acc_meter.count > 0 else 0.0
        total_pill_acc = pill_acc_meter.avg if pill_acc_meter.count > 0 else 0.0
    
    return {
        'loss': total_loss,
        'script_acc': total_script_acc,
        'pill_acc': total_pill_acc
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    embedding_model: nn.Module,
    val_loader: DataLoader,
    config: TrainingConfig,
    ddp_manager: DDPManager,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Validation loop."""
    
    model.eval()
    embedding_model.eval()
    
    # Metrics
    loss_meter = AverageMeter('Loss')
    script_acc_meter = AverageMeter('Script Acc')
    pill_acc_meter = AverageMeter('Pill Acc')
    correct_script_acc_meter = AverageMeter('Correct Script Acc')
    incorrect_script_acc_meter = AverageMeter('Incorrect Script Acc')
    
    # Progress bar
    if ddp_manager.is_main_process():
        pbar = tqdm(
            total=len(val_loader),
            desc=f'Val {epoch}',
            dynamic_ncols=True
        )
    
    for batch in val_loader:
        images = batch['images'].to(ddp_manager.device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(ddp_manager.device, non_blocking=True)
        expected_ndcs = batch['expected_ndcs'].to(ddp_manager.device, non_blocking=True)
        expected_counts = batch['expected_counts'].to(ddp_manager.device, non_blocking=True)
        expected_mask = batch['expected_mask'].to(ddp_manager.device, non_blocking=True)
        is_correct = batch['is_correct'].to(ddp_manager.device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(ddp_manager.device, non_blocking=True)
        
        B, max_pills, C, H, W = images.shape
        
        with autocast(enabled=config.use_amp):
            # Extract embeddings
            images_flat = images.view(B * max_pills, C, H, W)
            embeddings_flat, logits_flat = embedding_model(images_flat)
            pill_embeddings = embeddings_flat.view(B, max_pills, -1)
            pill_logits = logits_flat.view(B, max_pills, -1)
            
            pill_probs = F.softmax(pill_logits, dim=-1)
            pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
            
            # Forward
            script_logits, pill_anomaly = model(
                pill_embeddings=pill_embeddings,
                pill_confidences=pill_confidences,
                prescription_ndcs=expected_ndcs,
                prescription_counts=expected_counts.unsqueeze(-1),
                pill_mask=pill_mask,
                prescription_mask=expected_mask
            )
            
            # Loss
            loss, _ = compute_loss(
                script_logits, pill_anomaly,
                is_correct, pill_correct, pill_mask,
                config.script_loss_weight, config.pill_loss_weight
            )
        
        # Metrics
        metrics = compute_metrics(
            script_logits, pill_anomaly,
            is_correct, pill_correct, pill_mask
        )
        
        loss_meter.update(loss.item(), B)
        script_acc_meter.update(metrics['script_acc'], B)
        pill_acc_meter.update(metrics['pill_acc'], B)
        correct_script_acc_meter.update(metrics['correct_script_acc'], B)
        incorrect_script_acc_meter.update(metrics['incorrect_script_acc'], B)
        
        if ddp_manager.is_main_process():
            pbar.set_postfix({
                'val_loss': f'{loss_meter.avg:.4f}',
                'val_acc': f'{script_acc_meter.avg:.4f}',
                'script_acc': f'{script_acc_meter.avg:.4f}',
                'pill_acc': f'{pill_acc_meter.avg:.4f}'
            })
            pbar.update(1)
    
    if ddp_manager.is_main_process():
        pbar.close()
    
    # Synchronize
    if ddp_manager.is_distributed:
        metrics_tensor = torch.tensor([
            loss_meter.sum, loss_meter.count,
            script_acc_meter.sum, script_acc_meter.count,
            pill_acc_meter.sum, pill_acc_meter.count
        ], device=ddp_manager.device, dtype=torch.float64)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Safe division with epsilon to prevent division by zero
        eps = 1e-8
        loss_count = max(metrics_tensor[1].item(), eps)
        script_count = max(metrics_tensor[3].item(), eps)
        pill_count = max(metrics_tensor[5].item(), eps)
        
        total_loss = metrics_tensor[0].item() / loss_count
        total_script_acc = metrics_tensor[2].item() / script_count
        total_pill_acc = metrics_tensor[4].item() / pill_count
    else:
        total_loss = loss_meter.avg if loss_meter.count > 0 else 0.0
        total_script_acc = script_acc_meter.avg if script_acc_meter.count > 0 else 0.0
        total_pill_acc = pill_acc_meter.avg if pill_acc_meter.count > 0 else 0.0
    
    return {
        'loss': total_loss,
        'script_acc': total_script_acc,
        'pill_acc': total_pill_acc,
        'correct_script_acc': correct_script_acc_meter.avg if correct_script_acc_meter.count > 0 else 0.0,
        'incorrect_script_acc': incorrect_script_acc_meter.avg if incorrect_script_acc_meter.count > 0 else 0.0
    }


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Prescription-Aware Pill Verifier')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--index-file', type=str, default='dataset_index.csv', help='Index file name')
    parser.add_argument('--build-index', action='store_true', help='Build index and exit')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of NDC classes')
    parser.add_argument('--resnet-checkpoint', type=str, default=None, help='Pretrained ResNet checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--error-rate', type=float, default=0.5, help='Synthetic error rate')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        index_file=args.index_file,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        error_rate=args.error_rate,
        resume_from=args.resume,
        seed=args.seed
    )
    
    # Build index if requested
    if args.build_index:
        index_path = os.path.join(args.data_dir, config.index_file)
        DatasetIndexer.build_index(args.data_dir, index_path)
        print(f"Index saved to {index_path}")
        return
    
    # Initialize DDP
    ddp_manager = DDPManager(config)
    ddp_manager.setup()
    
    # Setup logging
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logging(ddp_manager.rank, config.output_dir)
    
    # Set seeds for reproducibility
    random.seed(config.seed + ddp_manager.rank)
    np.random.seed(config.seed + ddp_manager.rank)
    torch.manual_seed(config.seed + ddp_manager.rank)
    torch.cuda.manual_seed_all(config.seed + ddp_manager.rank)
    
    # Enable cuDNN benchmarking for A100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 for A100 (faster matmul)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logger.info(f"Training config: {config}")
    
    # Load index
    index_path = os.path.join(config.data_dir, config.index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index file not found: {index_path}\n"
            f"Run with --build-index first"
        )
    
    logger.info(f"Loading index from {index_path}")
    data_by_split, class_to_idx = DatasetIndexer.load_index(index_path)
    
    config.num_classes = len(class_to_idx)
    logger.info(f"Found {config.num_classes} NDC classes")
    
    # Create datasets
    train_dataset = PrescriptionDataset(
        data_dir=config.data_dir,
        split_data=data_by_split['train'],
        class_to_idx=class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=config.error_rate,
        is_training=True
    )
    
    val_dataset = PrescriptionDataset(
        data_dir=config.data_dir,
        split_data=data_by_split['valid'],
        class_to_idx=class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=0.5,  # Fixed error rate for validation
        is_training=False
    )
    
    logger.info(f"Train prescriptions: {len(train_dataset)}")
    logger.info(f"Val prescriptions: {len(val_dataset)}")
    
    # Validate we have enough data
    if len(train_dataset) == 0:
        raise ValueError("No training prescriptions found! Check your data directory and index file.")
    if len(val_dataset) == 0:
        raise ValueError("No validation prescriptions found! Check your data directory and index file.")
    
    # Ensure we have enough data for all GPUs
    if ddp_manager.is_distributed:
        min_samples_needed = ddp_manager.world_size * config.batch_size
        if len(train_dataset) < min_samples_needed:
            logger.warning(
                f"Training dataset ({len(train_dataset)}) is smaller than "
                f"world_size * batch_size ({min_samples_needed}). "
                f"Some GPUs may receive no data!"
            )
        if len(val_dataset) < ddp_manager.world_size:
            logger.warning(
                f"Validation dataset ({len(val_dataset)}) is smaller than "
                f"world_size ({ddp_manager.world_size}). "
                f"Some GPUs may receive no data during validation!"
            )
    
    # Create samplers
    if ddp_manager.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ddp_manager.world_size,
            rank=ddp_manager.rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=ddp_manager.world_size,
            rank=ddp_manager.rank,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=prescription_collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=prescription_collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Create models
    embedding_model = EmbeddingExtractor(
        pretrained_path=args.resnet_checkpoint,
        num_classes=config.num_classes,
        freeze=True
    ).to(ddp_manager.device)
    
    verifier = PrescriptionAwareVerifier(
        pill_embed_dim=embedding_model.embed_dim,
        ndc_vocab_size=config.num_classes,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    ).to(ddp_manager.device)
    
    # Wrap with DDP
    if ddp_manager.is_distributed:
        verifier = DDP(
            verifier,
            device_ids=[ddp_manager.local_rank],
            output_device=ddp_manager.local_rank,
            find_unused_parameters=config.find_unused_parameters,
            broadcast_buffers=True
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in verifier.parameters())
    trainable_params = sum(p.numel() for p in verifier.parameters() if p.requires_grad)
    logger.info(f"Verifier parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        verifier.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=config.use_amp)
    
    # Learning rate scheduler
    lr_scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=config.min_lr
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_script_acc = 0.0
    
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        start_epoch, best_script_acc = load_checkpoint(
            config.resume_from,
            verifier, embedding_model, optimizer, scaler,
            ddp_manager.device
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}, best_script_acc={best_script_acc:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.epochs):
        # Set epoch for distributed sampler
        if ddp_manager.is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Update learning rate
        current_lr = lr_scheduler.step(epoch)
        logger.info(f"\nEpoch {epoch} | LR: {current_lr:.6f}")
        
        # Train
        train_metrics = train_epoch(
            verifier, embedding_model, train_loader,
            optimizer, scaler, config, ddp_manager, epoch, logger
        )
        
        logger.info(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Script Acc: {train_metrics['script_acc']:.4f} | "
            f"Pill Acc: {train_metrics['pill_acc']:.4f}"
        )
        
        # Validate
        val_metrics = validate(
            verifier, embedding_model, val_loader,
            config, ddp_manager, epoch, logger
        )
        
        logger.info(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Script Acc: {val_metrics['script_acc']:.4f} | "
            f"Pill Acc: {val_metrics['pill_acc']:.4f}"
        )
        
        # Check for best model
        is_best = val_metrics['script_acc'] > best_script_acc
        if is_best:
            best_script_acc = val_metrics['script_acc']
            logger.info(f"New best script accuracy: {best_script_acc:.4f}")
        
        # Save checkpoint (rank 0 only)
        if ddp_manager.is_main_process():
            if (epoch + 1) % config.save_every_n_epochs == 0 or is_best:
                save_checkpoint(
                    verifier, embedding_model, optimizer, scaler,
                    epoch, best_script_acc, config, config.output_dir, is_best
                )
            
            # Create validation collage
            if (epoch + 1) % config.collage_every_n_epochs == 0:
                collage_path = os.path.join(
                    config.output_dir, f'collage_epoch_{epoch}.png'
                )
                create_validation_collage(
                    verifier.module if hasattr(verifier, 'module') else verifier,
                    embedding_model,
                    val_loader,
                    ddp_manager.device,
                    collage_path,
                    num_samples=config.num_collage_samples,
                    epoch=epoch
                )
                logger.info(f"Saved validation collage to {collage_path}")
        
        # Synchronize before next epoch
        ddp_manager.barrier()
    
    logger.info(f"\nTraining complete! Best script accuracy: {best_script_acc:.4f}")
    
    # Cleanup
    ddp_manager.cleanup()


if __name__ == '__main__':
    main()