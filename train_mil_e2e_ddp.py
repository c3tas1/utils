import os
import glob
import argparse
import random
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
# With 8 A100s, we can be aggressive.
# Global Batch Size = BATCH_SIZE_PER_GPU * 8
TRAIN_BAG_SIZE = 32    # Sample 32 pills per bag (Better context)
BATCH_SIZE_PER_GPU = 8 # 8 bags per GPU * 8 GPUs = 64 Bags per step
EPOCHS = 20

# Learning Rates
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3

# --- DDP SETUP ---
def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --- MODEL ---
class EndToEndMIL(nn.Module):
    def __init__(self, num_classes, backbone_path=None):
        super().__init__()
        
        # 1. Backbone
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Load your 98% accurate weights
        if backbone_path:
            if is_main_process():
                print(f"    Loading pre-trained weights from {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Remove FC weights
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            base_model.load_state_dict(new_state_dict, strict=False)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. MIL Head
        self.input_dim = 512
        self.attention_V = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        # x: [Batch, Bag, 3, 224, 224]
        batch_size, bag_size, C, H, W = x.shape
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract
        features = self.feature_extractor(x)
        features = features.view(batch_size, bag_size, -1)
        
        # Attention
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        # Classify
        bag_embedding = torch.sum(features * A, dim=1)
        logits = self.bag_classifier(bag_embedding)
        return logits, A

# --- DATASET ---
class BagDataset(Dataset):
    def __init__(self, root_dir, bag_size, transform, is_train=True):
        self.bag_size = bag_size
        self.transform = transform
        self.is_train = is_train
        self.bags = []
        
        # Only scan on Rank 0 to prevent filesystem thrashing, then broadcast?
        # For simplicity, all scan (it's fast enough)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if is_main_process():
            print(f"--> Scanning {root_dir}...")
            
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            sub_items = os.listdir(cls_folder)
            has_subfolders = any(os.path.isdir(os.path.join(cls_folder, i)) for i in sub_items)
            
            if has_subfolders:
                bags = [os.path.join(cls_folder, d) for d in sub_items if os.path.isdir(os.path.join(cls_folder, d))]
                for b in bags:
                    imgs = glob.glob(os.path.join(b, "*.jpg"))
                    if imgs: self.bags.append((imgs, self.class_to_idx[cls]))
            else:
                imgs = glob.glob(os.path.join(cls_folder, "*.jpg"))
                if imgs: self.bags.append((imgs, self.class_to_idx[cls]))

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        image_paths, label = self.bags[idx]
        
        if len(image_paths) > self.bag_size:
            if self.is_train:
                selected_paths = random.sample(image_paths, self.bag_size)
            else:
                # Deterministic sampling for val
                random.seed(42 + idx) 
                selected_paths = random.sample(image_paths, self.bag_size)
        else:
            selected_paths = image_paths
            while len(selected_paths) < self.bag_size:
                selected_paths += image_paths[:self.bag_size - len(selected_paths)]
        
        images = []
        for p in selected_paths:
            img = Image.open(p).convert('RGB')
            images.append(self.transform(img))
            
        return torch.stack(images), torch.tensor(label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--backbone_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="mil_end_to_end_ddp.pth")
    args = parser.parse_args()
    
    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f"--> Training End-to-End on {torch.cuda.device_count()} GPUs")

    # 1. Transforms
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
    
    # 2. Datasets & Distributed Samplers
    train_ds = BagDataset(args.train_dir, TRAIN_BAG_SIZE, train_tfm, is_train=True)
    val_ds = BagDataset(args.val_dir, TRAIN_BAG_SIZE, val_tfm, is_train=False) # Use same bag size for val
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler, 
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler, 
        num_workers=8, pin_memory=True
    )
    
    # 3. Model & DDP Wrapper
    model = EndToEndMIL(len(train_ds.classes), args.backbone_path).to(device)
    # SyncBatchNorm ensures Batch Norm stats are accurate across GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    
    # 4. Optimizer
    optimizer = optim.AdamW([
        {'params': model.module.feature_extractor.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.bag_classifier.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_V.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_U.parameters(), 'lr': LR_HEAD},
        {'params': model.module.attention_weights.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        # Local metrics
        local_loss = torch.tensor(0.0).to(device)
        local_correct = torch.tensor(0.0).to(device)
        local_total = torch.tensor(0.0).to(device)
        
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            pbar = train_loader
            
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(imgs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            local_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            local_correct += (preds == labels).sum()
            local_total += labels.size(0)
            
            if is_main_process() and isinstance(pbar, tqdm):
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Aggregating Validation
        model.eval()
        val_correct = torch.tensor(0.0).to(device)
        val_total = torch.tensor(0.0).to(device)
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum()
                val_total += labels.size(0)
        
        # SUM up results from all GPUs
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
        
        if is_main_process():
            val_acc = (val_correct / val_total).item()
            print(f"--> [Epoch {epoch+1}] Val Acc: {val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'classes': train_ds.classes
                }, args.save_path)
                print(f"--> Saved Best Model ({best_acc:.2%})")
                
    cleanup_ddp()

if __name__ == "__main__":
    main()
