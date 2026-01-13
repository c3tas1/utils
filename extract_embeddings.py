#!/usr/bin/env python3
"""
Fast Parallel Embedding Extraction
===================================

Uses ALL 8 GPUs to extract embeddings in parallel.
Should take ~30-60 minutes instead of 29 hours!

Usage:
    torchrun --nproc_per_node=8 extract_embeddings_fast.py \
        --data-dir /path/to/data \
        --backbone output_phase1/backbone_best.pt \
        --output output_phase2/embeddings_cache.pkl
"""

import os
import sys
import csv
import argparse
import pickle
import tempfile
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PillClassifier(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet34()
        self.backbone.fc = torch.nn.Linear(512, num_classes)
    
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
        embeddings = torch.flatten(x, 1)
        logits = self.backbone.fc(embeddings)
        return embeddings, logits


class ImagePathDataset(Dataset):
    """Simple dataset that loads images by path."""
    
    def __init__(self, data_dir: str, paths: list, input_size: int = 224):
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
            img = self.transform(img)
            return img, path, True
        except Exception as e:
            # Return dummy image for failed loads
            return torch.zeros(3, 224, 224), path, False


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    paths = [b[1] for b in batch]
    valid = [b[2] for b in batch]
    return images, paths, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--input-size', type=int, default=224)
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
    
    if rank == 0:
        print("=" * 60)
        print("FAST PARALLEL EMBEDDING EXTRACTION")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Batch size: {args.batch_size} Ã {world_size} = {args.batch_size * world_size}")
        print(f"Device: {device}")
    
    # Load backbone
    checkpoint = torch.load(args.backbone, map_location=device)
    num_classes = checkpoint['num_classes']
    class_to_idx = checkpoint['class_to_idx']
    
    backbone = PillClassifier(num_classes).to(device)
    backbone.load_state_dict(checkpoint['model'])
    backbone.eval()
    
    if rank == 0:
        print(f"Loaded backbone with {num_classes} classes")
    
    # Load all unique image paths
    index_file = os.path.join(args.data_dir, 'dataset_index.csv')
    all_paths = set()
    
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_paths.add(row['relative_path'])
    
    all_paths = sorted(list(all_paths))
    
    if rank == 0:
        print(f"Total unique images: {len(all_paths):,}")
        print(f"Images per GPU: ~{len(all_paths) // world_size:,}")
    
    # Create dataset and distributed sampler
    dataset = ImagePathDataset(args.data_dir, all_paths, args.input_size)
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    # Extract embeddings
    local_embeddings = {}
    local_logits = {}
    failed_count = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f'GPU {rank}', total=len(loader))
    else:
        pbar = loader
    
    with torch.no_grad():
        for images, paths, valid_flags in pbar:
            images = images.to(device, non_blocking=True)
            
            with autocast():
                emb, log = backbone.get_embeddings_and_logits(images)
            
            emb = emb.cpu()
            log = log.cpu()
            
            for i, (path, is_valid) in enumerate(zip(paths, valid_flags)):
                if is_valid:
                    local_embeddings[path] = emb[i]
                    local_logits[path] = log[i]
                else:
                    failed_count += 1
            
            if rank == 0:
                pbar.set_postfix({'extracted': len(local_embeddings), 'failed': failed_count})
    
    if rank == 0:
        print(f"\nGPU {rank}: Extracted {len(local_embeddings):,} embeddings, {failed_count} failed")
    
    # Gather from all GPUs
    if is_distributed:
        # Save local results to temp file
        temp_dir = os.path.dirname(args.output)
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'embeddings_rank_{rank}.pkl')
        
        with open(temp_file, 'wb') as f:
            pickle.dump({
                'embeddings': local_embeddings,
                'logits': local_logits
            }, f)
        
        if rank == 0:
            print(f"Saved local embeddings to {temp_file}")
        
        # Sync all processes
        dist.barrier()
        
        # Rank 0 merges all files
        if rank == 0:
            print("\nMerging embeddings from all GPUs...")
            all_embeddings = {}
            all_logits = {}
            
            for r in range(world_size):
                temp_file = os.path.join(temp_dir, f'embeddings_rank_{r}.pkl')
                print(f"  Loading from GPU {r}...")
                
                with open(temp_file, 'rb') as f:
                    data = pickle.load(f)
                
                all_embeddings.update(data['embeddings'])
                all_logits.update(data['logits'])
                
                # Clean up temp file
                os.remove(temp_file)
            
            print(f"\nTotal embeddings: {len(all_embeddings):,}")
            
            # Save final merged file
            print(f"Saving to {args.output}...")
            with open(args.output, 'wb') as f:
                pickle.dump({
                    'embeddings': all_embeddings,
                    'logits': all_logits,
                    'class_to_idx': class_to_idx
                }, f)
            
            print("â Done!")
    else:
        # Single GPU - save directly
        print(f"Saving to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump({
                'embeddings': local_embeddings,
                'logits': local_logits,
                'class_to_idx': class_to_idx
            }, f)
        print("â Done!")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
