import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image
import torch.nn.functional as F

# Import architecture
from model_utils import GatedAttentionMIL

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85  # Only flag foreign pill if model is 85% sure it's wrong

# --- 1. SPECIAL DATASET FOR RX FOLDERS ---
class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bags = [] # List of (prescription_id, [list_of_image_paths])
        
        # Expectation: root_dir/Prescription_ID/image.jpg
        print(f"--> Indexing prescriptions in {root_dir}...")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for rx_id in tqdm(subdirs):
            rx_path = os.path.join(root_dir, rx_id)
            # Gather all jpg images in this prescription folder
            img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
            if len(img_paths) > 0:
                self.bags.append((rx_id, img_paths))
                
        print(f"--> Found {len(self.bags)} prescriptions.")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        rx_id, file_paths = self.bags[idx]
        images = []
        valid_paths = []
        
        for p in file_paths:
            try:
                with Image.open(p) as img_raw:
                    img = img_raw.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                    valid_paths.append(p)
            except: continue
        
        if len(images) == 0:
            return torch.zeros((1, 3, 224, 224)), rx_id, []
            
        return torch.stack(images), rx_id, valid_paths

# --- 2. COLLATE TO HANDLE FILENAMES ---
def collate_inference(batch):
    # Batch is list of (tensor_stack, rx_id, file_paths)
    batch_images, rx_ids, batch_paths = zip(*batch)
    
    # Pad images for batch processing
    max_pills = max([x.shape[0] for x in batch_images])
    c, h, w = batch_images[0].shape[1:]
    
    padded_imgs = []
    masks = []
    
    for stack in batch_images:
        n = stack.shape[0]
        pad_size = max_pills - n
        # Mask: 1=Real Pill, 0=Padding
        m = torch.cat([torch.ones(n), torch.zeros(pad_size)])
        masks.append(m)
        
        if pad_size > 0:
            padding = torch.zeros((pad_size, c, h, w))
            padded_imgs.append(torch.cat([stack, padding], dim=0))
        else:
            padded_imgs.append(stack)
            
    return torch.stack(padded_imgs), torch.stack(masks), rx_ids, batch_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to folder containing Rx_ID subfolders")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to ORIGINAL TRAIN DIR (needed to load class names correctly)")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. Get Class Names from Training Directory (CRITICAL for mapping)
    classes = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
    print(f"--> Loaded {len(classes)} classes from training dir: {classes}")

    # 2. Setup Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = InferenceDataset(args.test_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_inference, num_workers=4)

    # 3. Load Model
    model = GatedAttentionMIL(num_classes=len(classes)).to(DEVICE)
    chk = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = chk['model_state_dict'] if 'model_state_dict' in chk else chk
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    foreign_pills_log = []
    prescriptions_log = []

    print(f"--> Scanning for foreign objects (Threshold: {CONFIDENCE_THRESHOLD*100}% confidence)...")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, mask, rx_ids, batch_paths in tqdm(loader):
                images = images.to(DEVICE)
                mask = mask.to(DEVICE)
                
                # Forward Pass
                bag_logits, inst_logits, _ = model(images, mask)
                
                # A. PREDICT BAG (The Prescription Label)
                bag_probs = F.softmax(bag_logits, dim=1)
                bag_conf, bag_preds = torch.max(bag_probs, dim=1)
                
                # B. CHECK INSTANCES (The Foreign Pill Hunt)
                B, M, _ = inst_logits.shape
                inst_probs = F.softmax(inst_logits, dim=2) # [Batch, Pills, Classes]
                
                for b in range(B):
                    rx_id = rx_ids[b]
                    prediction_name = classes[bag_preds[b]]
                    prediction_conf = bag_conf[b].item()
                    file_list = batch_paths[b]
                    
                    # Store Bag Result
                    prescriptions_log.append({
                        "Rx_ID": rx_id,
                        "Predicted_Drug": prediction_name,
                        "Confidence": f"{prediction_conf:.4f}"
                    })
                    
                    # --- THE FOREIGN PILL LOGIC ---
                    # We look at every pill in this specific bag
                    num_pills = int(mask[b].sum().item())
                    
                    for p_idx in range(num_pills):
                        # Get this pill's individual prediction
                        p_prob = inst_probs[b, p_idx]
                        p_conf, p_pred_idx = torch.max(p_prob, dim=0)
                        
                        # Logic:
                        # 1. Pill Prediction != Bag Prediction
                        # 2. Pill Confidence > Threshold (Ignore blurry/glare/back-of-pill low confidence errors)
                        if p_pred_idx != bag_preds[b] and p_conf > CONFIDENCE_THRESHOLD:
                            
                            foreign_class_name = classes[p_pred_idx]
                            bad_file_path = file_list[p_idx] # Exact filename
                            
                            print(f"\n[ALERT] Foreign Pill in {rx_id}!")
                            print(f"        File: {os.path.basename(bad_file_path)}")
                            print(f"        Detected as: {foreign_class_name} ({p_conf:.1%})")
                            print(f"        Prescription is: {prediction_name}")
                            
                            foreign_pills_log.append({
                                "Prescription_ID": rx_id,
                                "Foreign_File_Name": os.path.basename(bad_file_path),
                                "Detected_As": foreign_class_name,
                                "Confidence": f"{p_conf.item():.4f}",
                                "Context_Prescription_Label": prediction_name
                            })

    # Save Results
    pd.DataFrame(prescriptions_log).to_csv("inference_predictions.csv", index=False)
    
    if len(foreign_pills_log) > 0:
        df_foreign = pd.DataFrame(foreign_pills_log)
        df_foreign.to_csv("FOREIGN_OBJECT_REPORT.csv", index=False)
        print("\n" + "="*50)
        print(f"WARNING: {len(foreign_pills_log)} FOREIGN OBJECTS DETECTED")
        print("Report saved to: FOREIGN_OBJECT_REPORT.csv")
        print("="*50)
    else:
        print("\nSUCCESS: No foreign objects detected.")

if __name__ == "__main__":
    main()
