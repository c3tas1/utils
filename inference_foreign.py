import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image

# Import architecture
from model_utils import GatedAttentionMIL

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Threshold: If model is >85% sure a pill is different from the rest, flag it.
CONFIDENCE_THRESHOLD = 0.85 

# --- 1. DATASET THAT PARSES GT FROM FOLDER NAME ---
class InferenceDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.bags = [] 
        
        print(f"--> Indexing prescriptions in {root_dir}...")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        skipped = 0
        for folder_name in tqdm(subdirs):
            # --- PARSING LOGIC ---
            # User Rule: "Amoxicillin_Rx1" -> split('_')[0] -> "Amoxicillin"
            try:
                gt_class_str = folder_name.split('_')[0]
                
                # Check if this class exists in our training map
                if gt_class_str not in self.class_to_idx:
                    # Optional: Print warning if class is unknown
                    # print(f"Warning: Unknown class '{gt_class_str}' in folder '{folder_name}'. Skipping.")
                    skipped += 1
                    continue
                    
                gt_label = self.class_to_idx[gt_class_str]
                rx_path = os.path.join(root_dir, folder_name)
                img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
                
                if len(img_paths) > 0:
                    self.bags.append({
                        "rx_id": folder_name,
                        "paths": img_paths,
                        "label": gt_label,
                        "gt_str": gt_class_str
                    })
            except Exception as e:
                print(f"Error parsing {folder_name}: {e}")
                
        print(f"--> Indexed {len(self.bags)} valid prescriptions (Skipped {skipped} unknown classes).")

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
            return torch.zeros((1, 3, 224, 224)), -1, item["rx_id"], []
            
        return torch.stack(images), item["label"], item["rx_id"], valid_paths

# --- 2. COLLATE FUNCTION ---
def collate_inference(batch):
    # Unpack
    batch_images, labels, rx_ids, batch_paths = zip(*batch)
    
    # Pad images
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
            
    return torch.stack(padded_imgs), torch.stack(masks), torch.tensor(labels), rx_ids, batch_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Folder with Rx_ID subfolders")
    parser.add_argument("--train_dir", type=str, required=True, help="Original Train Dir (to get class mapping)")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Class Map
    classes = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"--> Loaded {len(classes)} classes.")

    # 2. Setup Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = InferenceDataset(args.test_dir, class_to_idx, transform=tfm)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_inference, num_workers=4)

    # 3. Load Model
    model = GatedAttentionMIL(num_classes=len(classes)).to(DEVICE)
    chk = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = chk['model_state_dict'] if 'model_state_dict' in chk else chk
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Metrics
    correct = 0
    total = 0
    
    foreign_log = []
    accuracy_log = []

    print(f"--> Running Inference...")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, mask, labels, rx_ids, batch_paths in tqdm(loader):
                images = images.to(DEVICE)
                mask = mask.to(DEVICE)
                labels = labels.to(DEVICE) # Now we have Ground Truth labels!
                
                # Model Inference
                bag_logits, inst_logits, _ = model(images, mask)
                
                # --- A. BAG ACCURACY ---
                bag_probs = F.softmax(bag_logits, dim=1)
                bag_conf, bag_preds = torch.max(bag_probs, dim=1)
                
                correct += (bag_preds == labels).sum().item()
                total += labels.size(0)
                
                # --- B. FOREIGN PILL DETECTION ---
                B, M, _ = inst_logits.shape
                inst_probs = F.softmax(inst_logits, dim=2)
                
                for b in range(B):
                    curr_rx = rx_ids[b]
                    curr_pred = classes[bag_preds[b]]
                    curr_gt = classes[labels[b]]
                    curr_files = batch_paths[b]
                    
                    # Log Accuracy Data
                    accuracy_log.append({
                        "Prescription_ID": curr_rx,
                        "Ground_Truth": curr_gt,
                        "Prediction": curr_pred,
                        "Correct": (curr_pred == curr_gt),
                        "Confidence": f"{bag_conf[b].item():.4f}"
                    })
                    
                    # Check Instances
                    num_pills = int(mask[b].sum().item())
                    for p in range(num_pills):
                        p_conf, p_idx = torch.max(inst_probs[b, p], dim=0)
                        
                        # LOGIC: 
                        # If Pill Prediction != Bag Prediction
                        # AND Confidence > 85% (Ignore blurry/ambiguous pills)
                        if p_idx != bag_preds[b] and p_conf > CONFIDENCE_THRESHOLD:
                            
                            foreign_class = classes[p_idx]
                            fname = os.path.basename(curr_files[p])
                            
                            # Console Alert
                            # print(f"\n[ALERT] Foreign Object in {curr_rx}")
                            # print(f"        File: {fname}")
                            # print(f"        Detected: {foreign_class} ({p_conf:.1%})")
                            
                            foreign_log.append({
                                "Prescription_ID": curr_rx,
                                "File_Name": fname,
                                "Detected_Class": foreign_class,
                                "Confidence": f"{p_conf.item():.4f}",
                                "Bag_Prediction": curr_pred
                            })

    # --- FINAL REPORT ---
    acc = correct / total if total > 0 else 0
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {acc:.2%}")
    print(f"Foreign Objects Found: {len(foreign_log)}")
    print("="*40)
    
    # Save CSVs
    pd.DataFrame(accuracy_log).to_csv("final_accuracy_report.csv", index=False)
    if len(foreign_log) > 0:
        pd.DataFrame(foreign_log).to_csv("FOREIGN_OBJECTS_FOUND.csv", index=False)
        print("--> Saved 'FOREIGN_OBJECTS_FOUND.csv'")
    
    print("--> Saved 'final_accuracy_report.csv'")

if __name__ == "__main__":
    main()
