import os
import glob
import argparse
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Batch Size: 8 GPUs * 8 Bags/GPU * 32 Pills/Bag = ~2048 Images per step
TRAIN_BAG_SIZE = 32    
BATCH_SIZE_PER_GPU = 8 
EPOCHS = 20

# Differential Learning Rates
LR_BACKBONE = 1e-5  # Slow fine-tuning for the ResNet
LR_HEAD = 1e-3      # Fast learning for the MIL Head

# --- DDP UTILS ---
def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --- MODEL ARCHITECTURE ---
class EndToEndMIL(nn.Module):
    def __init__(self, num_classes, backbone_path=None):
        super().__init__()
        
        # 1. Backbone (ResNet34)
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Optional: Load your pre-trained weights (skipping FC layer)
        if backbone_path:
            if is_main_process():
                print(f"    Loading pre-trained weights from {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Remove head weights to avoid shape mismatch
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            base_model.load_state_dict(new_state_dict, strict=False)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. MIL Heads
        self.input_dim = 512
        
        # Attention Mechanism
        self.attention_V = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        # Classifiers
        self.bag_classifier = nn.Linear(self.input_dim, num_classes)
        self.instance_classifier = nn.Linear(self.input_dim, num_classes) # For Pill Accuracy

    def forward(self, x):
        # x: [Batch, Bag, 3, 224, 224]
        batch_size, bag_size, C, H, W = x.shape
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract Features
        features = self.feature_extractor(x)
        features = features.view(batch_size, bag_size, -1) # [Batch, Bag, 512]
        
        # A. Instance Predictions (Pill Level)
        flat_features = features.view(-1, self.input_dim)
        instance_logits = self.instance_classifier(flat_features) 
        instance_logits = instance_logits.view(batch_size, bag_size, -1)
        
        # B. Bag Predictions (MIL Level)
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        bag_embedding = torch.sum(features * A, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A

# --- SMART DATASET (Parses {id}_patch_{n}) ---
class BagDataset(Dataset):
    def __init__(self, root_dir, bag_size, transform, is_train=True):
        self.bag_size = bag_size
        self.transform = transform
        self.bags = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if is_main_process():
            print(f"--> Scanning {root_dir}...")

        total_real_bags = 0
        
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            cls_idx = self.class_to_idx[cls]
            
            # Collect all jpgs
            all_files = [f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')]
            
            # Group by Prefix: {className}_{prescriptionId}
            # Expected format: Amoxicillin_Rx55_patch_1.jpg
            bag_groups = {}
            
            for f in all_files:
                # Extract Bag ID (everything before '_patch_')
                if "_patch_" in f:
                    bag_id = f.split("_patch_")[0]
                else:
                    # Robust fallback: Use filename itself if pattern breaks
                    bag_id = f 
                
                if bag_id not in bag_groups:
                    bag_groups[bag_id] = []
                
                bag_groups[bag_id].append(os.path.join(cls_folder, f))
            
            total_real_bags += len(bag_groups)
            
            # Create chunks from grouped bags
            for bag_id, image_paths in bag_groups.items():
                self._create_chunks(image_paths, cls_idx, is_train)

        if is_main_process():
            print(f"    Found {total_real_bags} unique prescriptions.")
            print(f"    Created {len(self.bags)} training batches (size {self.bag_size}).")

    def _create_chunks(self, image_paths, label, is_train):
        # Shuffle internally if training (mixes patches within the SAME bag)
        if is_train:
            random.shuffle(image_paths)
        else:
            image_paths.sort()
            
        num_imgs = len(image_paths)
        if num_imgs == 0: return

        # Split into chunks of 32
        num_chunks = math.ceil(num_imgs / self.bag_size)
        
        for i in range(num_chunks):
            chunk = image_paths[i*self.bag_size : (i+1)*self.bag_size]
            
            # Padding: Repeat internal images if chunk is small
            if len(chunk) < self.bag_size:
                while len(chunk) < self.bag_size:
                    chunk += chunk[:self.bag_size - len(chunk)]
                chunk = chunk[:self.bag_size]
            
            self.bags.append((chunk, label))

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        image_paths, label = self.bags[idx]
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(self.transform(img))
            except:
                # Black image if corrupted
                images.append(torch.zeros(3, 224, 224))
        return torch.stack(images), torch.tensor(label)

# --- TRAINING LOOP ---
def run_epoch(mode, model, loader, criterion, optimizer, device, epoch):
    is_train = (mode == 'train')
    if is_train: model.train()
    else: model.eval()

    # Metrics: [loss, bag_correct, bag_total, pill_correct, pill_total]
    metrics = torch.zeros(5).to(device) 
    
    if is_main_process():
        pbar = tqdm(loader, desc=f"{mode.capitalize()} {epoch+1}")
    else:
        pbar = loader
        
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if is_train: optimizer.zero_grad()
            
        with torch.set_grad_enabled(is_train):
            bag_logits, inst_logits, _ = model(imgs)
            
            # 1. Bag Loss
            bag_loss = criterion(bag_logits, labels)
            
            # 2. Pill Loss (Instance Supervision)
            B, N, C = inst_logits.shape
            inst_labels = labels.unsqueeze(1).expand(B, N).reshape(-1)
            inst_logits_flat = inst_logits.reshape(-1, C)
            inst_loss = criterion(inst_logits_flat, inst_labels)
            
            # Dual Loss
            total_loss = bag_loss + 0.5 * inst_loss
            
            if is_train:
                total_loss.backward()
                optimizer.step()
        
        # Calculate Accuracy
        bag_preds = torch.argmax(bag_logits, dim=1)
        bag_corr = (bag_preds == labels).sum()
        
        inst_preds = torch.argmax(inst_logits_flat, dim=1)
        pill_corr = (inst_preds == inst_labels).sum()
        
        metrics[0] += total_loss.item()
        metrics[1] += bag_corr
        metrics[2] += labels.size(0)
        metrics[3] += pill_corr
        metrics[4] += inst_labels.size(0)
        
        if is_main_process() and isinstance(pbar, tqdm) and is_train:
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

    # Reduce across GPUs
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0] / len(loader)
    bag_acc = metrics[1] / metrics[2]
    pill_acc = metrics[3] / metrics[4]
    
    return avg_loss.item(), bag_acc.item(), pill_acc.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--backbone_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="mil_end_to_end_final.pth")
    args = parser.parse_args()
    
    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f"--> Training End-to-End on {torch.cuda.device_count()} GPUs")

    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_ds = BagDataset(args.train_dir, TRAIN_BAG_SIZE, train_tfm, is_train=True)
    val_ds = BagDataset(args.val_dir, TRAIN_BAG_SIZE, val_tfm, is_train=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler, num_workers=8, pin_memory=True)
    
    # Model Setup
    model = EndToEndMIL(len(train_ds.classes), args.backbone_path).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.module.feature_extractor.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.bag_classifier.parameters(), 'lr': LR_HEAD},
        {'params': model.module.instance_classifier.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_V.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_U.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_weights.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        
        t_loss, t_bag_acc, t_pill_acc = run_epoch('train', model, train_loader, criterion, optimizer, device, epoch)
        v_loss, v_bag_acc, v_pill_acc = run_epoch('val', model, val_loader, criterion, optimizer, device, epoch)
        
        scheduler.step()
        
        if is_main_process():
            print(f"\n--> [Epoch {epoch+1}/{EPOCHS}]")
            print(f"    Train Loss: {t_loss:.4f} | Bag Acc: {t_bag_acc:.2%} | Pill Acc: {t_pill_acc:.2%}")
            print(f"    Val   Loss: {v_loss:.4f} | Bag Acc: {v_bag_acc:.2%} | Pill Acc: {v_pill_acc:.2%}")
            
            if v_bag_acc > best_acc:
                best_acc = v_bag_acc
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'classes': train_ds.classes
                }, args.save_path)
                print(f"    --> Saved Best Model ({best_acc:.2%})")
            print("-" * 60)
                
    cleanup_ddp()

if __name__ == "__main__":
    main()
