import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
# Since data is just vectors, we can use massive batch sizes
BATCH_SIZE = 128  
EPOCHS = 30
LR = 0.001

class FeatureBagDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        # Sort classes to ensure consistent ordering
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        print(f"--> Indexing features in {root_dir}...")
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            idx = self.class_to_idx[cls]
            # Look for .pt files
            for pt_file in glob.glob(os.path.join(cls_folder, "*.pt")):
                self.files.append((pt_file, idx))
                
        print(f"--> Found {len(self.files)} prescriptions across {len(self.classes)} classes.")

    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        path, label = self.files[idx]
        # Load the feature tensor [N_pills, 512]
        features = torch.load(path) 
        return features, label, path

def collate_features(batch):
    features, labels, paths = zip(*batch)
    
    # 1. Determine max bag size in this batch for padding
    max_len = max([f.shape[0] for f in features])
    feat_dim = features[0].shape[1]
    
    padded_feats = []
    masks = []
    
    for f in features:
        n = f.shape[0]
        pad_size = max_len - n
        
        # Mask: 1 for Real Pill, 0 for Padding
        mask = torch.cat([torch.ones(n), torch.zeros(pad_size)])
        masks.append(mask)
        
        if pad_size > 0:
            padding = torch.zeros(pad_size, feat_dim)
            padded_feats.append(torch.cat([f, padding], dim=0))
        else:
            padded_feats.append(f)
            
    return torch.stack(padded_feats), torch.tensor(labels), torch.stack(masks), paths

class MILHead(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        
        # Gated Attention Mechanism
        self.attention_V = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        # Classifiers
        self.bag_classifier = nn.Linear(input_dim, num_classes)
        self.instance_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, mask):
        # x: [Batch, Pills, 512]
        
        # 1. Compute Attention Scores
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U) # [B, P, 1]
        
        # 2. Masking (Ignore padding)
        mask_expanded = mask.unsqueeze(-1).to(x.device)
        A = A.masked_fill(mask_expanded == 0, -1e9)
        A = torch.softmax(A, dim=1) 
        
        # 3. Aggregation (Weighted Sum of Pills)
        bag_embedding = torch.sum(x * A, dim=1) 
        
        # 4. Predictions
        bag_logits = self.bag_classifier(bag_embedding)
        instance_logits = self.instance_classifier(x) # Used for Foreign Object Detection
        
        return bag_logits, instance_logits, A

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to ./features_data/train")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to ./features_data/valid")
    parser.add_argument("--save_path", type=str, default="mil_final_model.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training on {device}")
    
    # 1. Data Setup
    train_ds = FeatureBagDataset(args.train_dir)
    val_ds = FeatureBagDataset(args.val_dir)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_features, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_features, shuffle=False, num_workers=4)
    
    # Check Feature Dimension from first file
    sample_feat, _, _ = train_ds[0]
    feat_dim = sample_feat.shape[1]
    print(f"--> Detected Feature Dimension: {feat_dim}")
    
    # 2. Model Setup
    model = MILHead(num_classes=len(train_ds.classes), input_dim=feat_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for features, labels, mask, _ in pbar:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            bag_logits, inst_logits, _ = model(features, mask)
            
            # Loss Calculation
            loss_bag = criterion(bag_logits, labels)
            
            # Optional: Instance Loss (Max Pooling) to help Foreign Object detection
            # We take the max probability instance and classify it as the bag label
            # This helps the model align "Strongest Pill" with "Bag Label"
            B, P, C = inst_logits.shape
            max_inst_logits, _ = torch.max(inst_logits, dim=1) # [B, C]
            loss_inst = criterion(max_inst_logits, labels)
            
            total_loss = loss_bag + (0.5 * loss_inst)
            
            total_loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += total_loss.item()
            corr, tot = calculate_accuracy(bag_logits, labels)
            train_correct += corr
            train_total += tot
            
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Acc": f"{train_correct/train_total:.1%}"})
            
        scheduler.step()
        
        # 4. Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels, mask, _ in val_loader:
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
                bag_logits, inst_logits, _ = model(features, mask)
                
                loss = criterion(bag_logits, labels)
                val_loss += loss.item()
                
                corr, tot = calculate_accuracy(bag_logits, labels)
                val_correct += corr
                val_total += tot
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print("-" * 60)
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_correct/train_total:.2%}")
        print(f"Valid Loss: {avg_val_loss:.4f} | Valid Acc: {val_acc:.2%}")
        print("-" * 60)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save dict with classes so inference is easy later
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_ds.classes,
                'input_dim': feat_dim
            }, args.save_path)
            print(f"--> Saved Best Model ({best_acc:.2%})")

    print("\n--> Training Complete. Final Model Saved.")

if __name__ == "__main__":
    main()
