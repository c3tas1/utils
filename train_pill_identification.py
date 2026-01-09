#!/usr/bin/env python3
"""
End-to-End Prescription-Aware Pill Verification Training
=========================================================
Trains the ENTIRE network: ResNet backbone + Verifier

Target Hardware: NVIDIA DGX A100 (8x A100 40GB GPUs)

Architecture:
    Images → ResNet34 (TRAINABLE) → Embeddings → Verifier → Outputs
    
Loss Functions:
    1. Script-level loss: Is the entire prescription correct?
    2. Pill anomaly loss: Which pills are wrong?
    3. Pill classification loss: What NDC is each pill? (helps ResNet)

Key Features:
    - End-to-end gradient flow through entire network
    - Multi-GPU DDP training with NCCL backend
    - Differential learning rates (lower for backbone, higher for new layers)
    - Gradient checkpointing for memory efficiency
    - Mixed precision (AMP) for speed
    - Robust error handling for A100 clusters

Usage:
    # Build index first
    python train_e2e_a100.py --build-index --data-dir /path/to/data
    
    # Train on 8 GPUs
    torchrun --nproc_per_node=8 train_e2e_a100.py \\
        --data-dir /path/to/data \\
        --output-dir ./output \\
        --epochs 100

Author: Claude (Anthropic)
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
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for end-to-end training on A100 cluster."""
    
    # Data paths
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Model architecture
    num_classes: int = 1000
    backbone: str = "resnet34"  # resnet34 or resnet50
    pill_embed_dim: int = 512   # 512 for resnet34, 2048 for resnet50
    verifier_hidden_dim: int = 256
    verifier_num_heads: int = 8
    verifier_num_layers: int = 4
    verifier_dropout: float = 0.1
    
    # Prescription constraints
    max_pills_per_prescription: int = 200
    min_pills_per_prescription: int = 5
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 2  # Per GPU - smaller because training backbone
    gradient_accumulation_steps: int = 4  # Effective batch = 2 * 4 * 8 GPUs = 64
    
    # Learning rates (differential)
    backbone_lr: float = 1e-5      # Lower for pretrained backbone
    classifier_lr: float = 1e-4    # Medium for classifier head
    verifier_lr: float = 1e-4      # Higher for new verifier
    weight_decay: float = 0.01
    
    # LR schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Loss weights
    script_loss_weight: float = 1.0
    pill_anomaly_loss_weight: float = 0.5
    pill_classification_loss_weight: float = 0.3  # Helps backbone learn
    
    # Synthetic error injection
    error_rate: float = 0.5
    
    # A100 optimizations
    use_amp: bool = True
    use_gradient_checkpointing: bool = True  # Critical for memory
    max_grad_norm: float = 1.0
    
    # DDP settings
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = 30
    find_unused_parameters: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    validate_every_n_epochs: int = 1
    
    # Logging
    log_every_n_steps: int = 10
    collage_every_n_epochs: int = 5
    num_collage_samples: int = 16
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None


# =============================================================================
# Logging
# =============================================================================

def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """Setup logging - only rank 0 logs to console and file."""
    logger = logging.getLogger("E2E-PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers = []  # Clear existing handlers
    
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DDP Manager
# =============================================================================

class DDPManager:
    """Manages Distributed Data Parallel with robust error handling for A100."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        self.device = torch.device('cuda:0')
    
    def setup(self) -> None:
        """Initialize DDP."""
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
        
        # NCCL optimizations for A100
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['NCCL_IB_DISABLE'] = '0'
        os.environ['NCCL_P2P_DISABLE'] = '0'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        
        timeout = timedelta(minutes=self.config.dist_timeout_minutes)
        
        dist.init_process_group(
            backend=self.config.dist_backend,
            timeout=timeout,
            rank=self.rank,
            world_size=self.world_size
        )
        
        dist.barrier()
        
        if self.rank == 0:
            print(f"DDP initialized: {self.world_size} GPUs")
    
    def cleanup(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()


@contextmanager
def ddp_sync_context(model: nn.Module, sync: bool = True):
    """Control gradient synchronization for accumulation."""
    if isinstance(model, DDP) and not sync:
        with model.no_sync():
            yield
    else:
        yield


# =============================================================================
# Dataset Indexing
# =============================================================================

class DatasetIndexer:
    """Fast CSV-based indexing."""
    
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
                            file_ndc, rx_id, patch_no = match.groups()
                            rel_path = str(img_path.relative_to(data_dir))
                            writer.writerow([split, ndc, rx_id, int(patch_no), rel_path])
                            stats[f'{split}_images'] += 1
                
                stats[f'{split}_ndcs'] = len(ndc_dirs)
        
        # Count prescriptions
        rx_per_split = defaultdict(set)
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rx_per_split[row['split']].add(row['prescription_id'])
        
        for split in splits:
            stats[f'{split}_prescriptions'] = len(rx_per_split[split])
        
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
        transform=None,
        max_pills: int = 200,
        min_pills: int = 5,
        error_rate: float = 0.0,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
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
    
    def _group_by_prescription(self, data: List[dict]) -> Dict[str, List[dict]]:
        prescriptions = defaultdict(list)
        for item in data:
            prescriptions[item['prescription_id']].append(item)
        
        return {
            rx_id: sorted(pills, key=lambda x: x['patch_no'])
            for rx_id, pills in prescriptions.items()
            if self.min_pills <= len(pills) <= self.max_pills
        }
    
    def _build_pill_library(self) -> Dict[str, List[str]]:
        library = defaultdict(list)
        for pills in self.prescriptions.values():
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_image(self, rel_path: str) -> Image.Image:
        try:
            img = Image.open(self.data_dir / rel_path)
            img.load()
            return img.convert('RGB')
        except:
            return Image.new('RGB', (224, 224), (0, 0, 0))
    
    def _get_wrong_pill(self, correct_ndc: str) -> Tuple[str, str, int]:
        available = [n for n in self.pill_library if n != correct_ndc]
        if not available:
            wrong_ndc = correct_ndc
        else:
            wrong_ndc = random.choice(available)
        wrong_path = random.choice(self.pill_library[wrong_ndc])
        return wrong_path, wrong_ndc, self.class_to_idx[wrong_ndc]
    
    def _inject_errors(self, pills: List[dict]) -> Tuple[List[dict], List[int], str]:
        error_type = random.choices(
            ['single', 'few', 'many', 'all'],
            weights=[0.5, 0.3, 0.15, 0.05]
        )[0]
        
        n = len(pills)
        modified = [p.copy() for p in pills]
        
        if error_type == 'single':
            n_wrong = 1
        elif error_type == 'few':
            n_wrong = random.randint(2, min(5, max(2, n // 4)))
        elif error_type == 'many':
            n_wrong = random.randint(max(2, n // 4), max(3, n // 2))
        else:
            n_wrong = n
        
        wrong_indices = random.sample(range(n), min(n_wrong, n))
        
        for idx in wrong_indices:
            orig_ndc = modified[idx]['ndc']
            wrong_path, wrong_ndc, _ = self._get_wrong_pill(orig_ndc)
            modified[idx] = {
                'relative_path': wrong_path,
                'ndc': wrong_ndc,
                'patch_no': modified[idx]['patch_no'],
                'prescription_id': modified[idx]['prescription_id']
            }
        
        return modified, wrong_indices, error_type
    
    def get_prescription_composition(self, rx_id: str) -> Dict[str, int]:
        pills = self.prescriptions[rx_id]
        comp = defaultdict(int)
        for p in pills:
            comp[p['ndc']] += 1
        return dict(comp)
    
    def __len__(self):
        return len(self.prescription_ids)
    
    def __getitem__(self, idx):
        rx_id = self.prescription_ids[idx]
        original_pills = self.prescriptions[rx_id]
        
        inject_error = self.is_training and random.random() < self.error_rate
        
        if inject_error:
            pills, wrong_indices, _ = self._inject_errors(original_pills)
            is_correct = False
        else:
            pills = original_pills
            wrong_indices = []
            is_correct = True
        
        images = []
        labels = []
        
        for pill in pills:
            img = self._load_image(pill['relative_path'])
            if self.transform:
                img = self.transform(img)
            images.append(img)
            labels.append(self.class_to_idx[pill['ndc']])
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Expected composition (original, not modified)
        comp = self.get_prescription_composition(rx_id)
        expected_ndcs = [self.class_to_idx[n] for n in comp.keys()]
        expected_counts = list(comp.values())
        
        # Per-pill correctness
        pill_correct = torch.ones(len(pills), dtype=torch.float)
        for i in wrong_indices:
            pill_correct[i] = 0.0
        
        return {
            'images': images,
            'labels': labels,
            'num_pills': len(pills),
            'expected_ndcs': torch.tensor(expected_ndcs, dtype=torch.long),
            'expected_counts': torch.tensor(expected_counts, dtype=torch.float),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': pill_correct,
            'wrong_indices': wrong_indices,
            'prescription_id': rx_id
        }


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate for variable-length prescriptions."""
    max_pills = max(b['num_pills'] for b in batch)
    max_expected = max(len(b['expected_ndcs']) for b in batch)
    B = len(batch)
    C, H, W = batch[0]['images'].shape[1:]
    
    images = torch.zeros(B, max_pills, C, H, W)
    labels = torch.full((B, max_pills), -100, dtype=torch.long)  # -100 for ignore
    pill_correct = torch.ones(B, max_pills)
    pill_mask = torch.ones(B, max_pills, dtype=torch.bool)
    
    expected_ndcs = torch.zeros(B, max_expected, dtype=torch.long)
    expected_counts = torch.zeros(B, max_expected, dtype=torch.float)
    expected_mask = torch.ones(B, max_expected, dtype=torch.bool)
    
    is_correct = torch.zeros(B)
    num_pills = torch.zeros(B, dtype=torch.long)
    rx_ids = []
    wrong_indices = []
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        n_exp = len(b['expected_ndcs'])
        
        images[i, :n] = b['images']
        labels[i, :n] = b['labels']
        pill_correct[i, :n] = b['pill_correct']
        pill_mask[i, :n] = False
        
        expected_ndcs[i, :n_exp] = b['expected_ndcs']
        expected_counts[i, :n_exp] = b['expected_counts']
        expected_mask[i, :n_exp] = False
        
        is_correct[i] = b['is_correct']
        num_pills[i] = n
        rx_ids.append(b['prescription_id'])
        wrong_indices.append(b['wrong_indices'])
    
    return {
        'images': images,
        'labels': labels,
        'pill_correct': pill_correct,
        'pill_mask': pill_mask,
        'expected_ndcs': expected_ndcs,
        'expected_counts': expected_counts,
        'expected_mask': expected_mask,
        'is_correct': is_correct,
        'num_pills': num_pills,
        'prescription_ids': rx_ids,
        'wrong_indices': wrong_indices
    }


# =============================================================================
# Model: End-to-End Trainable
# =============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet backbone with gradient checkpointing support.
    Outputs embeddings and classification logits.
    """
    
    def __init__(
        self,
        model_name: str = "resnet34",
        num_classes: int = 1000,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Load pretrained model
        if model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.embed_dim = 512
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.embed_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Split into stages for gradient checkpointing
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
    
    def _forward_stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def _forward_layer1(self, x):
        return self.layer1(x)
    
    def _forward_layer2(self, x):
        return self.layer2(x)
    
    def _forward_layer3(self, x):
        return self.layer3(x)
    
    def _forward_layer4(self, x):
        return self.layer4(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) or (B, N, C, H, W)
        Returns:
            embeddings: (B, embed_dim) or (B, N, embed_dim)
            logits: (B, num_classes) or (B, N, num_classes)
        """
        original_shape = x.shape
        
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
        
        # Forward with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(self._forward_stem, x, use_reentrant=False)
            x = checkpoint(self._forward_layer1, x, use_reentrant=False)
            x = checkpoint(self._forward_layer2, x, use_reentrant=False)
            x = checkpoint(self._forward_layer3, x, use_reentrant=False)
            x = checkpoint(self._forward_layer4, x, use_reentrant=False)
        else:
            x = self._forward_stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        x = self.avgpool(x)
        embeddings = x.flatten(1)
        logits = self.classifier(embeddings)
        
        if len(original_shape) == 5:
            embeddings = embeddings.view(B, N, -1)
            logits = logits.view(B, N, -1)
        
        return embeddings, logits


class PrescriptionVerifier(nn.Module):
    """Prescription-aware verification head."""
    
    def __init__(
        self,
        pill_embed_dim: int = 512,
        ndc_vocab_size: int = 1000,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # NDC embedding
        self.ndc_embedding = nn.Embedding(ndc_vocab_size, hidden_dim)
        
        # Prescription encoder
        self.rx_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        
        # Pill projector
        self.pill_projector = nn.Sequential(
            nn.Linear(pill_embed_dim + 1, hidden_dim),  # +1 for confidence
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(2)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        
        # Pill self-attention
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
        
        # Summary token and pooling
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.script_head = nn.Sequential(
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        pill_embeddings: torch.Tensor,
        pill_confidences: torch.Tensor,
        rx_ndcs: torch.Tensor,
        rx_counts: torch.Tensor,
        pill_mask: Optional[torch.Tensor] = None,
        rx_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = pill_embeddings.size(0)
        
        # Encode prescription
        rx_embed = self.ndc_embedding(rx_ndcs)
        count_scale = rx_counts / (rx_counts.sum(dim=1, keepdim=True) + 1e-8)
        rx_embed = rx_embed + count_scale.unsqueeze(-1) * 0.1
        
        if self.use_gradient_checkpointing and self.training:
            rx_encoded = checkpoint(
                lambda x: self.rx_encoder(x, src_key_padding_mask=rx_mask),
                rx_embed,
                use_reentrant=False
            )
        else:
            rx_encoded = self.rx_encoder(rx_embed, src_key_padding_mask=rx_mask)
        
        # Project pill features
        pill_features = torch.cat([pill_embeddings, pill_confidences], dim=-1)
        pill_proj = self.pill_projector(pill_features)
        
        # Cross-attention
        for i in range(len(self.cross_attn)):
            attn_out, _ = self.cross_attn[i](pill_proj, rx_encoded, rx_encoded, key_padding_mask=rx_mask)
            pill_proj = self.cross_norms[i](pill_proj + attn_out)
        
        # Self-attention among pills
        if self.use_gradient_checkpointing and self.training:
            pill_encoded = checkpoint(
                lambda x: self.pill_encoder(x, src_key_padding_mask=pill_mask),
                pill_proj,
                use_reentrant=False
            )
        else:
            pill_encoded = self.pill_encoder(pill_proj, src_key_padding_mask=pill_mask)
        
        # Per-pill anomaly
        pill_anomaly = self.pill_anomaly_head(pill_encoded).squeeze(-1)
        
        # Script-level prediction
        summary = self.summary_token.expand(B, -1, -1)
        pooled, _ = self.pool_attn(summary, pill_encoded, pill_encoded, key_padding_mask=pill_mask)
        pooled = self.pool_norm(summary + pooled)
        
        script_logit = self.script_head(pooled.squeeze(1))
        
        return script_logit, pill_anomaly


class EndToEndModel(nn.Module):
    """
    Complete end-to-end model combining backbone and verifier.
    """
    
    def __init__(
        self,
        backbone: ResNetBackbone,
        verifier: PrescriptionVerifier
    ):
        super().__init__()
        self.backbone = backbone
        self.verifier = verifier
    
    def forward(
        self,
        images: torch.Tensor,
        rx_ndcs: torch.Tensor,
        rx_counts: torch.Tensor,
        pill_mask: Optional[torch.Tensor] = None,
        rx_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, N_pills, C, H, W)
            rx_ndcs: (B, N_rx)
            rx_counts: (B, N_rx)
            pill_mask: (B, N_pills) - True for padding
            rx_mask: (B, N_rx) - True for padding
        
        Returns:
            script_logit: (B, 1)
            pill_anomaly: (B, N_pills)
            pill_logits: (B, N_pills, num_classes)
        """
        # Get embeddings and classification from backbone
        embeddings, pill_logits = self.backbone(images)
        
        # Compute confidences
        pill_probs = F.softmax(pill_logits, dim=-1)
        pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
        
        # Verifier
        script_logit, pill_anomaly = self.verifier(
            embeddings, pill_confidences,
            rx_ndcs, rx_counts,
            pill_mask, rx_mask
        )
        
        return script_logit, pill_anomaly, pill_logits


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        if n > 0:
            self.sum += val * n
            self.count += n
    
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


class MultiLRScheduler:
    """Scheduler with different LRs for different parameter groups."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch: int):
        for i, (pg, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg['lr'] = lr
        
        return [pg['lr'] for pg in self.optimizer.param_groups]


def compute_losses(
    script_logit: torch.Tensor,
    pill_anomaly: torch.Tensor,
    pill_logits: torch.Tensor,
    is_correct: torch.Tensor,
    pill_correct: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor,
    config: TrainingConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute all three losses:
    1. Script-level loss
    2. Pill anomaly loss
    3. Pill classification loss
    """
    
    # 1. Script loss
    script_loss = F.binary_cross_entropy_with_logits(
        script_logit.squeeze(-1), is_correct
    )
    
    # 2. Pill anomaly loss
    pill_anomaly_target = 1 - pill_correct  # 1 = anomaly
    pill_anomaly_loss_raw = F.binary_cross_entropy_with_logits(
        pill_anomaly, pill_anomaly_target, reduction='none'
    )
    valid_mask = ~pill_mask
    if valid_mask.any():
        pill_anomaly_loss = (pill_anomaly_loss_raw * valid_mask.float()).sum() / valid_mask.sum()
    else:
        pill_anomaly_loss = torch.tensor(0.0, device=script_loss.device)
    
    # 3. Pill classification loss (only for correct pills)
    B, N, C = pill_logits.shape
    pill_logits_flat = pill_logits.view(B * N, C)
    pill_labels_flat = pill_labels.view(B * N)
    
    # Use ignore_index for padding
    pill_cls_loss = F.cross_entropy(
        pill_logits_flat, pill_labels_flat, 
        ignore_index=-100, reduction='mean'
    )
    
    # Total loss
    total_loss = (
        config.script_loss_weight * script_loss +
        config.pill_anomaly_loss_weight * pill_anomaly_loss +
        config.pill_classification_loss_weight * pill_cls_loss
    )
    
    return total_loss, {
        'total': total_loss.item(),
        'script': script_loss.item(),
        'anomaly': pill_anomaly_loss.item(),
        'cls': pill_cls_loss.item()
    }


def compute_metrics(
    script_logit: torch.Tensor,
    pill_anomaly: torch.Tensor,
    pill_logits: torch.Tensor,
    is_correct: torch.Tensor,
    pill_correct: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor
) -> Dict[str, float]:
    """Compute accuracy metrics."""
    
    with torch.no_grad():
        # Script accuracy
        script_pred = (torch.sigmoid(script_logit.squeeze(-1)) > 0.5).float()
        script_acc = (script_pred == is_correct).float().mean().item()
        
        # Pill anomaly accuracy
        anomaly_pred = (torch.sigmoid(pill_anomaly) > 0.5).float()
        anomaly_target = 1 - pill_correct
        valid = ~pill_mask
        if valid.any():
            anomaly_acc = ((anomaly_pred == anomaly_target) & valid).float().sum() / valid.sum()
            anomaly_acc = anomaly_acc.item()
        else:
            anomaly_acc = 0.0
        
        # Pill classification accuracy (top-1)
        pill_pred = pill_logits.argmax(dim=-1)
        correct_cls = (pill_pred == pill_labels) & valid
        if valid.any():
            cls_acc = correct_cls.float().sum() / valid.sum()
            cls_acc = cls_acc.item()
        else:
            cls_acc = 0.0
    
    return {
        'script_acc': script_acc,
        'anomaly_acc': anomaly_acc,
        'cls_acc': cls_acc
    }


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: TrainingConfig,
    output_dir: str,
    is_best: bool = False
):
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
        'config': config.__dict__
    }
    
    path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, path)
    
    if is_best:
        torch.save(state, os.path.join(output_dir, 'best_model.pt'))
    
    # Cleanup old checkpoints
    checkpoints = sorted(
        [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    while len(checkpoints) > config.keep_last_n_checkpoints:
        os.remove(os.path.join(output_dir, checkpoints.pop(0)))


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler, device) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    
    return ckpt['epoch'], ckpt['best_metric']


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    ddp: DDPManager,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    
    model.train()
    
    # Loss meters
    total_loss_meter = AverageMeter()
    script_loss_meter = AverageMeter()
    anomaly_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()  # ResNet classification loss
    
    # Accuracy meters
    script_acc_meter = AverageMeter()
    anomaly_acc_meter = AverageMeter()
    cls_acc_meter = AverageMeter()  # Pill identification accuracy
    
    if ddp.is_main_process():
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}', dynamic_ncols=True)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        images = batch['images'].to(ddp.device, non_blocking=True)
        labels = batch['labels'].to(ddp.device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(ddp.device, non_blocking=True)
        rx_ndcs = batch['expected_ndcs'].to(ddp.device, non_blocking=True)
        rx_counts = batch['expected_counts'].to(ddp.device, non_blocking=True)
        rx_mask = batch['expected_mask'].to(ddp.device, non_blocking=True)
        is_correct = batch['is_correct'].to(ddp.device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(ddp.device, non_blocking=True)
        
        B = images.size(0)
        sync = (step + 1) % config.gradient_accumulation_steps == 0
        
        with ddp_sync_context(model, sync=sync):
            with autocast(enabled=config.use_amp):
                script_logit, pill_anomaly, pill_logits = model(
                    images, rx_ndcs, rx_counts, pill_mask, rx_mask
                )
                
                loss, loss_dict = compute_losses(
                    script_logit, pill_anomaly, pill_logits,
                    is_correct, pill_correct, labels, pill_mask,
                    config
                )
                loss = loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        if sync:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        metrics = compute_metrics(
            script_logit, pill_anomaly, pill_logits,
            is_correct, pill_correct, labels, pill_mask
        )
        
        # Update loss meters
        total_loss_meter.update(loss_dict['total'], B)
        script_loss_meter.update(loss_dict['script'], B)
        anomaly_loss_meter.update(loss_dict['anomaly'], B)
        cls_loss_meter.update(loss_dict['cls'], B)
        
        # Update accuracy meters
        script_acc_meter.update(metrics['script_acc'], B)
        anomaly_acc_meter.update(metrics['anomaly_acc'], B)
        cls_acc_meter.update(metrics['cls_acc'], B)
        
        if ddp.is_main_process():
            pbar.set_postfix({
                'loss': f'{total_loss_meter.avg:.4f}',
                'script_loss': f'{script_loss_meter.avg:.4f}',
                'resnet_loss': f'{cls_loss_meter.avg:.4f}',
                'script_acc': f'{script_acc_meter.avg:.4f}',
                'pill_acc': f'{cls_acc_meter.avg:.4f}'
            })
            pbar.update(1)
            
            # Log every N steps
            if (step + 1) % config.log_every_n_steps == 0:
                logger.info(
                    f"Epoch {epoch} Step {step+1}/{len(train_loader)} | "
                    f"Loss: {total_loss_meter.avg:.4f} | "
                    f"Script Loss: {script_loss_meter.avg:.4f} | "
                    f"ResNet Loss: {cls_loss_meter.avg:.4f} | "
                    f"Script Acc: {script_acc_meter.avg:.4f} | "
                    f"Pill Acc: {cls_acc_meter.avg:.4f}"
                )
    
    if ddp.is_main_process():
        pbar.close()
    
    # Sync metrics across GPUs
    if ddp.is_distributed:
        metrics_tensor = torch.tensor([
            total_loss_meter.sum, total_loss_meter.count,
            script_loss_meter.sum, script_loss_meter.count,
            cls_loss_meter.sum, cls_loss_meter.count,
            script_acc_meter.sum, script_acc_meter.count,
            cls_acc_meter.sum, cls_acc_meter.count
        ], device=ddp.device, dtype=torch.float64)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        return {
            'total_loss': metrics_tensor[0].item() / max(metrics_tensor[1].item(), 1),
            'script_loss': metrics_tensor[2].item() / max(metrics_tensor[3].item(), 1),
            'resnet_loss': metrics_tensor[4].item() / max(metrics_tensor[5].item(), 1),
            'script_acc': metrics_tensor[6].item() / max(metrics_tensor[7].item(), 1),
            'pill_acc': metrics_tensor[8].item() / max(metrics_tensor[9].item(), 1)
        }
    
    return {
        'total_loss': total_loss_meter.avg,
        'script_loss': script_loss_meter.avg,
        'resnet_loss': cls_loss_meter.avg,
        'script_acc': script_acc_meter.avg,
        'pill_acc': cls_acc_meter.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: TrainingConfig,
    ddp: DDPManager,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    
    model.eval()
    
    # Loss meters
    total_loss_meter = AverageMeter()
    script_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    
    # Accuracy meters
    script_acc_meter = AverageMeter()
    anomaly_acc_meter = AverageMeter()
    cls_acc_meter = AverageMeter()
    
    if ddp.is_main_process():
        pbar = tqdm(total=len(val_loader), desc=f'Val {epoch}', dynamic_ncols=True)
    
    for batch in val_loader:
        images = batch['images'].to(ddp.device, non_blocking=True)
        labels = batch['labels'].to(ddp.device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(ddp.device, non_blocking=True)
        rx_ndcs = batch['expected_ndcs'].to(ddp.device, non_blocking=True)
        rx_counts = batch['expected_counts'].to(ddp.device, non_blocking=True)
        rx_mask = batch['expected_mask'].to(ddp.device, non_blocking=True)
        is_correct = batch['is_correct'].to(ddp.device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(ddp.device, non_blocking=True)
        
        B = images.size(0)
        
        with autocast(enabled=config.use_amp):
            script_logit, pill_anomaly, pill_logits = model(
                images, rx_ndcs, rx_counts, pill_mask, rx_mask
            )
            
            loss, loss_dict = compute_losses(
                script_logit, pill_anomaly, pill_logits,
                is_correct, pill_correct, labels, pill_mask,
                config
            )
        
        metrics = compute_metrics(
            script_logit, pill_anomaly, pill_logits,
            is_correct, pill_correct, labels, pill_mask
        )
        
        # Update meters
        total_loss_meter.update(loss_dict['total'], B)
        script_loss_meter.update(loss_dict['script'], B)
        cls_loss_meter.update(loss_dict['cls'], B)
        
        script_acc_meter.update(metrics['script_acc'], B)
        anomaly_acc_meter.update(metrics['anomaly_acc'], B)
        cls_acc_meter.update(metrics['cls_acc'], B)
        
        if ddp.is_main_process():
            pbar.set_postfix({
                'loss': f'{total_loss_meter.avg:.4f}',
                'script_acc': f'{script_acc_meter.avg:.4f}',
                'pill_acc': f'{cls_acc_meter.avg:.4f}'
            })
            pbar.update(1)
    
    if ddp.is_main_process():
        pbar.close()
    
    # Sync
    if ddp.is_distributed:
        metrics_tensor = torch.tensor([
            total_loss_meter.sum, total_loss_meter.count,
            script_loss_meter.sum, script_loss_meter.count,
            cls_loss_meter.sum, cls_loss_meter.count,
            script_acc_meter.sum, script_acc_meter.count,
            cls_acc_meter.sum, cls_acc_meter.count
        ], device=ddp.device, dtype=torch.float64)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        return {
            'total_loss': metrics_tensor[0].item() / max(metrics_tensor[1].item(), 1),
            'script_loss': metrics_tensor[2].item() / max(metrics_tensor[3].item(), 1),
            'resnet_loss': metrics_tensor[4].item() / max(metrics_tensor[5].item(), 1),
            'script_acc': metrics_tensor[6].item() / max(metrics_tensor[7].item(), 1),
            'pill_acc': metrics_tensor[8].item() / max(metrics_tensor[9].item(), 1)
        }
    
    return {
        'total_loss': total_loss_meter.avg,
        'script_loss': script_loss_meter.avg,
        'resnet_loss': cls_loss_meter.avg,
        'script_acc': script_acc_meter.avg,
        'pill_acc': cls_acc_meter.avg
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='End-to-End Pill Verification Training')
    
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--build-index', action='store_true')
    
    parser.add_argument('--backbone', type=str, default='resnet34', choices=['resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--classifier-lr', type=float, default=1e-4)
    parser.add_argument('--verifier-lr', type=float, default=1e-4)
    
    parser.add_argument('--min-pills', type=int, default=5)
    parser.add_argument('--max-pills', type=int, default=200)
    parser.add_argument('--error-rate', type=float, default=0.5)
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
        verifier_lr=args.verifier_lr,
        min_pills_per_prescription=args.min_pills,
        max_pills_per_prescription=args.max_pills,
        error_rate=args.error_rate,
        resume_from=args.resume,
        seed=args.seed,
        num_workers=args.workers
    )
    
    # Build index
    if args.build_index:
        DatasetIndexer.build_index(args.data_dir, os.path.join(args.data_dir, config.index_file))
        return
    
    # DDP setup
    ddp = DDPManager(config)
    ddp.setup()
    
    # Logging
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logging(ddp.rank, config.output_dir)
    
    # Seeds
    random.seed(config.seed + ddp.rank)
    np.random.seed(config.seed + ddp.rank)
    torch.manual_seed(config.seed + ddp.rank)
    torch.cuda.manual_seed_all(config.seed + ddp.rank)
    
    # A100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    logger.info("=" * 70)
    logger.info("End-to-End Prescription Pill Verification Training")
    logger.info("=" * 70)
    logger.info(f"Device: {ddp.device}")
    logger.info(f"World size: {ddp.world_size}")
    
    # Load data
    index_path = os.path.join(config.data_dir, config.index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Run --build-index first: {index_path}")
    
    data_by_split, class_to_idx = DatasetIndexer.load_index(index_path)
    config.num_classes = len(class_to_idx)
    config.pill_embed_dim = 512 if config.backbone == 'resnet34' else 2048
    
    logger.info(f"Classes: {config.num_classes}")
    logger.info(f"Backbone: {config.backbone} (embed_dim={config.pill_embed_dim})")
    
    # Datasets
    train_dataset = PrescriptionDataset(
        config.data_dir, data_by_split['train'], class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=config.error_rate,
        is_training=True
    )
    
    val_dataset = PrescriptionDataset(
        config.data_dir, data_by_split['valid'], class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=0.5,
        is_training=False
    )
    
    logger.info(f"Train: {len(train_dataset)} prescriptions")
    logger.info(f"Val: {len(val_dataset)} prescriptions")
    
    if len(train_dataset) == 0:
        raise ValueError("No training data!")
    
    # Samplers and loaders
    if ddp.is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0
    )
    
    # Models
    backbone = ResNetBackbone(
        model_name=config.backbone,
        num_classes=config.num_classes,
        pretrained=True,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )
    
    verifier = PrescriptionVerifier(
        pill_embed_dim=config.pill_embed_dim,
        ndc_vocab_size=config.num_classes,
        hidden_dim=config.verifier_hidden_dim,
        num_heads=config.verifier_num_heads,
        num_layers=config.verifier_num_layers,
        dropout=config.verifier_dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )
    
    model = EndToEndModel(backbone, verifier).to(ddp.device)
    
    # Parameter groups with different LRs
    param_groups = [
        {
            'params': [p for n, p in model.backbone.named_parameters() 
                      if 'classifier' not in n and p.requires_grad],
            'lr': config.backbone_lr,
            'name': 'backbone'
        },
        {
            'params': model.backbone.classifier.parameters(),
            'lr': config.classifier_lr,
            'name': 'classifier'
        },
        {
            'params': model.verifier.parameters(),
            'lr': config.verifier_lr,
            'name': 'verifier'
        }
    ]
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # DDP wrap
    if ddp.is_distributed:
        model = DDP(
            model,
            device_ids=[ddp.local_rank],
            output_device=ddp.local_rank,
            find_unused_parameters=config.find_unused_parameters
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    scheduler = MultiLRScheduler(optimizer, config.warmup_epochs, config.epochs, config.min_lr)
    
    # Resume
    start_epoch = 0
    best_script_acc = 0.0
    
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        start_epoch, best_script_acc = load_checkpoint(
            config.resume_from, model, optimizer, scaler, ddp.device
        )
        start_epoch += 1
    
    # Effective batch size
    eff_batch = config.batch_size * config.gradient_accumulation_steps * ddp.world_size
    logger.info(f"Effective batch size: {config.batch_size} × {config.gradient_accumulation_steps} × {ddp.world_size} = {eff_batch}")
    logger.info(f"Learning rates: backbone={config.backbone_lr}, classifier={config.classifier_lr}, verifier={config.verifier_lr}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.epochs):
        if ddp.is_distributed:
            train_sampler.set_epoch(epoch)
        
        lrs = scheduler.step(epoch)
        logger.info(f"\nEpoch {epoch} | LRs: backbone={lrs[0]:.2e}, cls={lrs[1]:.2e}, ver={lrs[2]:.2e}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, config, ddp, epoch, logger
        )
        
        logger.info(
            f"Train | "
            f"Total Loss: {train_metrics['total_loss']:.4f} | "
            f"Script Loss: {train_metrics['script_loss']:.4f} | "
            f"ResNet Loss: {train_metrics['resnet_loss']:.4f} | "
            f"Script Acc: {train_metrics['script_acc']:.4f} | "
            f"Pill Acc: {train_metrics['pill_acc']:.4f}"
        )
        
        # Validate
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, config, ddp, epoch, logger)
            
            logger.info(
                f"Val   | "
                f"Total Loss: {val_metrics['total_loss']:.4f} | "
                f"Script Loss: {val_metrics['script_loss']:.4f} | "
                f"ResNet Loss: {val_metrics['resnet_loss']:.4f} | "
                f"Script Acc: {val_metrics['script_acc']:.4f} | "
                f"Pill Acc: {val_metrics['pill_acc']:.4f}"
            )
            
            is_best = val_metrics['script_acc'] > best_script_acc
            if is_best:
                best_script_acc = val_metrics['script_acc']
                logger.info(f"New best: {best_script_acc:.4f}")
            
            if ddp.is_main_process():
                if (epoch + 1) % config.save_every_n_epochs == 0 or is_best:
                    save_checkpoint(
                        model, optimizer, scaler, epoch,
                        best_script_acc, config, config.output_dir, is_best
                    )
        
        ddp.barrier()
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info(f"\nTraining complete! Best script accuracy: {best_script_acc:.4f}")
    ddp.cleanup()


if __name__ == '__main__':
    main()