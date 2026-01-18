import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
from PIL import Image

# Import architecture
from model_utils import GatedAttentionMIL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlignmentDataset(Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bags = [] 
        
        # Sort by length to ensure correct matching
        self.known_classes = sorted(class_list, key=len, reverse=True)
        
        # Only load FIRST 10 folders for rapid debugging
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirs = subdirs[:20] 
        
        print(f"--> Diagnostic Mode: Scanning first {len(subdirs)} folders only...")
        
        for folder_name in subdirs:
            found_class = None
            for cls in self.known_classes:
                if folder_name.startswith(cls):
                    found_class = cls
                    break
            
            if found_class:
                rx_path = os.path.join(root_dir, folder_name)
                img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
                if len(img_paths) > 0:
                    self.bags.append({
                        "rx_id": folder_name,
                        "gt_class": found_class,
                        "paths": img_paths
                    })

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        item = self.bags[idx]
        images = []
        for p in item["paths"]:
            try:
                with Image.open(p) as img_raw:
                    img = img_raw.convert('RGB')
                    if self.transform: img = self.transform(img)
                    images.append(img)
            except: continue
        
        if len(images) == 0: return torch.zeros((1, 3, 224, 224)), "Err", "Err"
        return torch.stack(images), item["rx_id"], item["gt_class"]

def collate_align(batch):
    batch_images, rx_ids, gt_classes = zip(*batch)
    max_pills = max([x.shape[0] for x in batch_images])
    c, h, w = batch_images[0].shape[1:]
    padded_imgs = []
    masks = []
    for stack in batch_images:
        n = stack.shape[0]
        pad_size = max_pills - n
        m = torch.cat([torch.ones(n), torch.zeros(pad_size)])
        masks.append(m)
        if pad_size > 0:
            padding = torch.zeros((pad_size, c, h, w))
            padded_imgs.append(torch.cat([stack, padding], dim=0))
        else:
            padded_imgs.append(stack)
    return torch.stack(padded_imgs), torch.stack(masks), rx_ids, gt_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Classes
    # CRITICAL: We print the first few to check sorting order
    raw_folders = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    train_classes = sorted(raw_folders)
    
    print("\n" + "="*60)
    print(" TRAINING CLASS INDEX MAP (First 5)")
    print("="*60)
    for i, c in enumerate(train_classes[:5]):
        print(f" Index {i}: {c}")
    print("="*60 + "\n")

    # 2. Setup
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = AlignmentDataset(args.test_dir, train_classes, transform=tfm)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_align, num_workers=0)

    # 3. Model
    model = GatedAttentionMIL(num_classes=len(train_classes)).to(DEVICE)
    chk = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = chk['model_state_dict'] if 'model_state_dict' in chk else chk
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("\n" + "="*60)
    print(f"{'GROUND TRUTH':<30} | {'PREDICTION':<30} | {'STATUS'}")
    print("="*60)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, mask, rx_ids, gt_classes in loader:
                images = images.to(DEVICE)
                mask = mask.to(DEVICE)
                
                bag_logits, _, _ = model(images, mask)
                bag_probs = F.softmax(bag_logits, dim=1)
                bag_conf, bag_preds = torch.max(bag_probs, dim=1)
                
                # Get the string name of the prediction
                pred_idx = bag_preds[0].item()
                pred_name = train_classes[pred_idx]
                gt_name = gt_classes[0]
                
                status = "✅ MATCH" if pred_name == gt_name else "❌ MISMATCH"
                print(f"{gt_name:<30} | {pred_name:<30} | {status}")

if __name__ == "__main__":
    main()
