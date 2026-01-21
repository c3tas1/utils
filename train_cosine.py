import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# --- CONFIG ---
# Batch size PER GPU. 
# Global Batch Size = 128 * 8 = 1024
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.0005 
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 2

# --- DDP HELPERS ---
def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()

def reduce_tensor(tensor):
    """Aggregates a metric (e.g., loss) across all GPUs for logging."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# --- DATASET ---
class FeatureBagDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Rank 0 logs the data finding, others stay silent
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"--> Indexing features in {root_dir}...")
            
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            idx = self.class_to_idx[cls]
            for pt_file in glob.glob(os.path.join(cls_folder, "*.pt")):
                self.files.append((pt_file, idx))

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        features = torch.load(path) # [N_pills, 512]
        return features, label

def collate_features(batch):
    features, labels = zip(*batch)
    max_len = max([f.shape[0] for f in features])
    feat_dim = features[0].shape[1]
    
    padded_feats = []
    masks = [] 
    
    for f in features:
        n = f.shape[0]
        pad_size = max_len - n
        mask = torch.cat([torch.ones(n), torch.zeros(pad_size)])
        masks.append(mask)
        if pad_size > 0:
            padded_feats.append(torch.cat([f, torch.zeros(pad_size, feat_dim)], dim=0))
        else:
            padded_feats.append(f)
            
    return torch.stack(padded_feats), torch.tensor(labels), torch.stack(masks)

# --- TRANSFORMER MODEL ---
class TransformerMIL(nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, 
            dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        h = self.relu(self.fc_in(x))
        key_padding_mask = (mask == 0) if mask is not None else None
        h_trans = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h_trans = self.norm(h_trans)
        
        attn_logits = self.attn_layer(h_trans)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        bag_embedding = torch.sum(h_trans * attn_weights, dim=1)
        logits = self.classifier(bag_embedding)
        return logits, h_trans, bag_embedding, attn_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="mil_transformer_ddp.pth")
    args = parser.parse_args()
    
    setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print(f"--> Training on {torch.cuda.device_count()} GPUs (DDP Mode)")

    # 1. Dataset & Sampler
    train_ds = FeatureBagDataset(args.train_dir)
    val_ds = FeatureBagDataset(args.val_dir)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, 
        collate_fn=collate_features, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, 
        collate_fn=collate_features, num_workers=4, pin_memory=True
    )
    
    # 2. Model Setup
    model = TransformerMIL(
        num_classes=len(train_ds.classes),
        hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS, n_heads=NUM_HEADS
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        running_loss = torch.tensor(0.0).to(device)
        correct = torch.tensor(0.0).to(device)
        total = torch.tensor(0.0).to(device)
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            pbar = train_loader
            
        for features, labels, mask in pbar:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            bag_logits, _, _, _ = model(features, mask)
            loss = criterion(bag_logits, labels)
            loss.backward()
            optimizer.step()
            
            # Local Metrics
            running_loss += loss.item()
            preds = torch.argmax(bag_logits, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
            
            # Update pbar on rank 0
            if local_rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 4. Global Validation (Aggregate across all GPUs)
        # We need to sum 'correct' and 'total' from all GPUs to get true accuracy
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        global_train_acc = correct / total
        
        # Validation Loop
        model.eval()
        val_correct = torch.tensor(0.0).to(device)
        val_total = torch.tensor(0.0).to(device)
        
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
                bag_logits, _, _, _ = model(features, mask)
                preds = torch.argmax(bag_logits, dim=1)
                val_correct += (preds == labels).sum()
                val_total += labels.size(0)
        
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
        global_val_acc = val_correct / val_total
        
        # 5. Logging & Saving (Rank 0 only)
        if local_rank == 0:
            print(f"--> [Epoch {epoch+1}] Train Acc: {global_train_acc:.2%} | Val Acc: {global_val_acc:.2%}")
            
            if global_val_acc > best_acc:
                best_acc = global_val_acc
                torch.save({
                    'model_state_dict': model.module.state_dict(), # Save underlying model, not DDP wrapper
                    'classes': train_ds.classes,
                    'hidden_dim': HIDDEN_DIM,
                    'n_layers': NUM_LAYERS,
                    'n_heads': NUM_HEADS
                }, args.save_path)
                print(f"--> Saved Best Model ({best_acc:.2%})")

    cleanup_ddp()

if __name__ == "__main__":
    main()
