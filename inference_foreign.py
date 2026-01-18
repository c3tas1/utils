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

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If model confidence for a specific pill is > 85% and it disagrees with the bag, flag it.
FOREIGN_THRESH = 0.85 

class SmartInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bags = [] 
        
        print(f"--> Scanning test directory: {root_dir}")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for folder_name in tqdm(subdirs):
            # LOGIC: "Amoxicillin_RxID_123" -> split('_')[0] -> "Amoxicillin"
            try:
                # 1. Extract Ground Truth from folder name
                gt_class_name = folder_name.split('_')[0]
                
                # 2. Collect images
                rx_path = os.path.join(root_dir, folder_name)
                img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
                
                if len(img_paths) > 0:
                    self.bags.append({
                        "rx_id": folder_name,      # "Amoxicillin_RxID_123"
                        "gt_class": gt_class_name, # "Amoxicillin"
                        "paths": img_paths
                    })
            except Exception as e:
                print(f"Skipping {folder_name}: {e}")
                
        print(f"--> Ready to test {len(self.bags)} prescriptions.")

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
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                    valid_paths.append(p)
            except: continue
        
        if len(images) == 0:
            # Handle empty folder edge case
            return torch.zeros((1, 3, 224, 224)), item["rx_id"], "Unknown", []
            
        return torch.stack(images), item["rx_id"], item["gt_class"], valid_paths

def collate_smart(batch):
    batch_images, rx_ids, gt_classes, batch_paths = zip(*batch)
    
    # Pad bags to same size for batching
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
    parser.add_argument("--test_dir", type=str, required=True, help="Folder with 'Class_RxID' subfolders")
    parser.add_argument("--train_dir", type=str, required=True, help="Original Train Dir (to get class order)")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Training Classes (To map Index -> Name)
    # The model outputs Index 0, 1, 2... we need to know 0="Amoxicillin"
    train_classes = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
    print(f"--> Model knows {len(train_classes)} drug classes: {train_classes}")

    # 2. Setup
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = SmartInferenceDataset(args.test_dir, transform=tfm)
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
                
                # Model Inference
                bag_logits, inst_logits, _ = model(images, mask)
                
                # Bag Predictions
                bag_probs = F.softmax(bag_logits, dim=1)
                bag_conf, bag_preds = torch.max(bag_probs, dim=1)
                
                # Instance Predictions (For Foreign Object)
                inst_probs = F.softmax(inst_logits, dim=2)
                
                B = images.shape[0]
                for b in range(B):
                    # A. CHECK BAG ACCURACY
                    pred_name = train_classes[bag_preds[b]]
                    ground_truth = gt_classes[b]
                    
                    if pred_name == ground_truth:
                        correct += 1
                    total += 1
                    
                    # B. FIND FOREIGN PILLS
                    num_pills = int(mask[b].sum().item())
                    for p in range(num_pills):
                        p_conf, p_idx = torch.max(inst_probs[b, p], dim=0)
                        p_pred_name = train_classes[p_idx]
                        
                        # --- THE LOGIC YOU ASKED FOR ---
                        # If Pill says "Ibuprofen" but Bag says "Amoxicillin"
                        # AND model is >85% confident
                        if p_pred_name != pred_name and p_conf > FOREIGN_THRESH:
                            
                            bad_file = os.path.basename(batch_paths[b][p])
                            
                            foreign_log.append({
                                "Prescription": rx_ids[b],
                                "Foreign_Patch_File": bad_file,  # <--- HERE IS THE FILE NAME
                                "Detected_As": p_pred_name,
                                "Confidence": f"{p_conf.item():.4f}",
                                "Main_Prescription_Is": pred_name
                            })

    # Final Report
    acc = correct / total if total > 0 else 0
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {acc:.2%}")
    print(f"Foreign Objects Detected: {len(foreign_log)}")
    print("="*40)
    
    if len(foreign_log) > 0:
        pd.DataFrame(foreign_log).to_csv("FOREIGN_OBJECTS_FOUND.csv", index=False)
        print("--> Detailed foreign object report saved to 'FOREIGN_OBJECTS_FOUND.csv'")

if __name__ == "__main__":
    main()
