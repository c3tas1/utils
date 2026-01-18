import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image

# Import your architecture
from model_utils import GatedAttentionMIL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOREIGN_THRESH = 0.85 

class FixedInferenceDataset(Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bags = [] 
        
        # Sort class list by length (descending) to ensure "Vitamin_C" matches before "Vitamin"
        self.known_classes = sorted(class_list, key=len, reverse=True)
        
        print(f"--> Scanning test directory: {root_dir}")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        matched_count = 0
        for i, folder_name in enumerate(tqdm(subdirs, desc="Indexing Folders")):
            # --- FIXED PARSING LOGIC ---
            found_class = None
            for cls in self.known_classes:
                # Check if folder starts with known class name
                if folder_name.startswith(cls):
                    found_class = cls
                    break
            
            if found_class:
                # --- VERIFICATION PRINT (First 5 matches only) ---
                if matched_count < 5:
                    print(f"   [DEBUG] Mapped Folder: '{folder_name}' --> Class: '{found_class}'")
                
                rx_path = os.path.join(root_dir, folder_name)
                img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
                
                if len(img_paths) > 0:
                    self.bags.append({
                        "rx_id": folder_name,
                        "gt_class": found_class,
                        "paths": img_paths
                    })
                    matched_count += 1
            else:
                # Print first few failures too, just in case
                if i < 5: 
                    print(f"   [WARNING] Could not map folder '{folder_name}' to any known class!")
                pass
                
        print(f"--> Successfully mapped {matched_count} prescriptions.")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        item = self.bags[idx]
        images = []
        valid_paths = []
        for p in item["paths"]:
            try:
                with Image.open(p) as img_raw:
                    img = img_raw.convert('RGB')
                    if self.transform: img = self.transform(img)
                    images.append(img)
                    valid_paths.append(p)
            except: continue
        
        if len(images) == 0:
            return torch.zeros((1, 3, 224, 224)), item["rx_id"], "Unknown", []
        return torch.stack(images), item["rx_id"], item["gt_class"], valid_paths

def collate_smart(batch):
    batch_images, rx_ids, gt_classes, batch_paths = zip(*batch)
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
    return torch.stack(padded_imgs), torch.stack(masks), rx_ids, gt_classes, batch_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Classes & PRINT THEM
    train_classes = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
    
    print("\n" + "="*50)
    print(f"CLASS VERIFICATION ({len(train_classes)} total)")
    print("="*50)
    print("First 5 classes detected:")
    for c in train_classes[:5]: print(f" - {c}")
    print("...")
    print("Last 5 classes detected:")
    for c in train_classes[-5:]: print(f" - {c}")
    print("="*50 + "\n")

    # 2. Setup Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = FixedInferenceDataset(args.test_dir, train_classes, transform=tfm)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_smart, num_workers=4)

    # 3. Load Model
    model = GatedAttentionMIL(num_classes=len(train_classes)).to(DEVICE)
    chk = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = chk['model_state_dict'] if 'model_state_dict' in chk else chk
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    correct = 0
    total = 0
    foreign_log = []

    print("--> Starting Analysis...")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, mask, rx_ids, gt_classes, batch_paths in tqdm(loader):
                images = images.to(DEVICE)
                mask = mask.to(DEVICE)
                
                bag_logits, inst_logits, _ = model(images, mask)
                bag_probs = F.softmax(bag_logits, dim=1)
                bag_conf, bag_preds = torch.max(bag_probs, dim=1)
                inst_probs = F.softmax(inst_logits, dim=2)
                
                B = images.shape[0]
                for b in range(B):
                    # Check Bag Accuracy
                    pred_name = train_classes[bag_preds[b]]
                    ground_truth = gt_classes[b]
                    
                    if pred_name == ground_truth:
                        correct += 1
                    total += 1
                    
                    # Check Foreign Pills
                    num_pills = int(mask[b].sum().item())
                    for p in range(num_pills):
                        p_conf, p_idx = torch.max(inst_probs[b, p], dim=0)
                        p_pred_name = train_classes[p_idx]
                        
                        if p_pred_name != pred_name and p_conf > FOREIGN_THRESH:
                            foreign_log.append({
                                "Prescription": rx_ids[b],
                                "Foreign_File": os.path.basename(batch_paths[b][p]),
                                "Detected_As": p_pred_name,
                                "Confidence": f"{p_conf.item():.4f}",
                                "Expected": pred_name
                            })

    acc = correct / total if total > 0 else 0
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {acc:.2%}")
    print(f"Foreign Objects Detected: {len(foreign_log)}")
    print("="*40)
    
    if len(foreign_log) > 0:
        pd.DataFrame(foreign_log).to_csv("FOREIGN_OBJECTS_FOUND.csv", index=False)

if __name__ == "__main__":
    main()
