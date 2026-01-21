import os
import glob
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Reduce this if you run out of GPU Memory (OOM)
TRAIN_BAG_SIZE = 16   # Number of pills to sample per bag during training
VAL_BAG_SIZE = 32     # Number of pills for validation
BATCH_SIZE = 4        # Number of BAGS per batch (Effective images = 4 * 16 = 64)
EPOCHS = 20

# Differential Learning Rates (Crucial!)
LR_BACKBONE = 1e-5    # Very slow learning for the pre-trained eye
LR_HEAD = 1e-3        # Fast learning for the brain

class EndToEndMIL(nn.Module):
    def __init__(self, num_classes, backbone_path=None):
        super().__init__()
        
        # 1. The Backbone (ResNet34)
        print("--> Initializing ResNet34 Backbone...")
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Optional: Load your 98% accurate weights if you have them
        if backbone_path:
            print(f"    Loading pre-trained weights from {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            # Clean keys
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Remove FC weights from loading (shape mismatch likely)
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            base_model.load_state_dict(new_state_dict, strict=False)

        # Remove the classification head
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. The MIL Head
        self.input_dim = 512
        self.attention_V = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch_Size, Bag_Size, 3, 224, 224]
        batch_size, bag_size, C, H, W = x.shape
        
        # Flatten to process images through ResNet: [Batch*Bag, 3, 224, 224]
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract Features
        features = self.feature_extractor(x) # [Batch*Bag, 512, 1, 1]
        features = features.view(batch_size, bag_size, -1) # [Batch, Bag, 512]
        
        # MIL Attention Mechanism
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U) # [Batch, Bag, 1]
        A = torch.softmax(A, dim=1)           # Normalize attention scores
        
        # Aggregate Features (Weighted Sum)
        bag_embedding = torch.sum(features * A, dim=1) # [Batch, 512]
        
        # Final Classification
        logits = self.bag_classifier(bag_embedding)
        return logits, A

class BagDataset(Dataset):
    def __init__(self, root_dir, bag_size, transform, is_train=True):
        self.bag_size = bag_size
        self.transform = transform
        self.is_train = is_train
        self.bags = []
        
        # Find classes
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        print(f"--> Scanning {root_dir}...")
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            # Each SUBFOLDER is a bag (e.g. Rx_001, Rx_002)
            # Assuming structure: Train/Amoxicillin/Rx_001/img1.jpg
            # If structure is Train/Amoxicillin/img1.jpg, we treat the CLASS folder as one giant bag (bad idea).
            # Let's assume structure: Train/Amoxicillin/img1.jpg, img2.jpg...
            # Since MIL usually requires distinct bags, if you have one folder per class, 
            # we will create "Artificial Bags" by grouping images randomly.
            
            # Check structure
            sub_items = os.listdir(cls_folder)
            has_subfolders = any(os.path.isdir(os.path.join(cls_folder, i)) for i in sub_items)
            
            if has_subfolders:
                # Proper MIL Structure: Class -> Bag -> Images
                bags = [os.path.join(cls_folder, d) for d in sub_items if os.path.isdir(os.path.join(cls_folder, d))]
                for b in bags:
                    imgs = glob.glob(os.path.join(b, "*.jpg"))
                    if imgs: self.bags.append((imgs, self.class_to_idx[cls]))
            else:
                # Flat Structure: Class -> Images
                # We will just treat the whole class as a pool and sample from it
                # Warning: This is "Weak Supervision" at its limit.
                imgs = glob.glob(os.path.join(cls_folder, "*.jpg"))
                if imgs: self.bags.append((imgs, self.class_to_idx[cls]))

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        image_paths, label = self.bags[idx]
        
        # SAMPLING STRATEGY
        if len(image_paths) > self.bag_size:
            # If training, random sample (Data Augmentation!)
            if self.is_train:
                selected_paths = random.sample(image_paths, self.bag_size)
            else:
                # If testing, take fixed sample (or first N) to be deterministic
                # Or better: random sample with fixed seed. Let's just take first N for speed.
                selected_paths = image_paths[:self.bag_size]
        else:
            # Padding (Loop over images if we don't have enough)
            selected_paths = image_paths
            while len(selected_paths) < self.bag_size:
                selected_paths += image_paths[:self.bag_size - len(selected_paths)]
        
        # Load Images
        images = []
        for p in selected_paths:
            img = Image.open(p).convert('RGB')
            images.append(self.transform(img))
            
        return torch.stack(images), torch.tensor(label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--backbone_path", type=str, default=None, help="Optional: Start from your 98% accurate backbone")
    parser.add_argument("--save_path", type=str, default="mil_end_to_end.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training End-to-End on {device}")
    
    # 1. Transforms
    # Strong augmentation for training helps MIL a lot!
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
    
    # 2. Datasets
    train_ds = BagDataset(args.train_dir, TRAIN_BAG_SIZE, train_tfm, is_train=True)
    val_ds = BagDataset(args.val_dir, VAL_BAG_SIZE, val_tfm, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. Model
    model = EndToEndMIL(len(train_ds.classes), args.backbone_path).to(device)
    
    # 4. Optimizer (Differential Learning Rates)
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': LR_BACKBONE}, # Slow
        {'params': model.bag_classifier.parameters(), 'lr': LR_HEAD},        # Fast
        {'params': model.attention_V.parameters(), 'lr': LR_HEAD},
        {'params': model.attention_U.parameters(), 'lr': LR_HEAD},
        {'params': model.attention_weights.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            # imgs: [Batch, Bag, 3, 224, 224]
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(imgs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.1%}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"--> Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_ds.classes
            }, args.save_path)
            print("--> Saved Best Model")

if __name__ == "__main__":
    main()
