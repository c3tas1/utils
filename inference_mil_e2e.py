import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import sys

# --- CONFIG ---
BATCH_SIZE = 32      # Process 32 images at a time
FOREIGN_THRESH = 0.90 # Only flag foreign object if model is 90% sure

# --- MODEL ARCHITECTURE (Must match training script) ---
class EndToEndMIL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone
        base_model = models.resnet34(weights=None) # Weights loaded from checkpoint later
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. Heads
        self.input_dim = 512
        self.attention_V = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        self.bag_classifier = nn.Linear(self.input_dim, num_classes)
        self.instance_classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        # x: [Batch, Bag, 3, 224, 224] or [N, 3, 224, 224] for inference
        if x.dim() == 4:
            x = x.unsqueeze(0) # Fake batch dim [1, N, C, H, W]
            
        batch_size, bag_size, C, H, W = x.shape
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract Features
        features = self.feature_extractor(x)
        features = features.view(batch_size, bag_size, -1) # [Batch, Bag, 512]
        
        # A. Instance Predictions
        flat_features = features.view(-1, self.input_dim)
        instance_logits = self.instance_classifier(flat_features) 
        instance_logits = instance_logits.view(batch_size, bag_size, -1)
        
        # B. Bag Predictions
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        bag_embedding = torch.sum(features * A, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A

def load_model(model_path, device):
    print(f"--> Loading End-to-End Model from {os.path.basename(model_path)}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Model file not found: {model_path}")

    # 1. Get Classes
    class_list = checkpoint.get('classes')
    if not class_list:
        sys.exit("[ERROR] Checkpoint missing 'classes' key. Use the correct training script.")
        
    # 2. Initialize Model
    model = EndToEndMIL(len(class_list)).to(device)
    
    # 3. Load State Dict (Handling DDP 'module.' prefix)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    return model, class_list

def predict_bag(img_paths, model, device, class_list, transform):
    img_paths = sorted(img_paths)
    
    # 1. Load Images into Batch
    images = []
    valid_paths = []
    
    for p in img_paths:
        try:
            with Image.open(p) as img:
                images.append(transform(img.convert('RGB')))
                valid_paths.append(p)
        except: pass
    
    if not images: return None
    
    # [N, 3, 224, 224]
    img_tensor = torch.stack(images).to(device)
    
    # 2. Run Inference
    # We pass the whole bag at once. If memory issues, we'd need to chunk it,
    # but inference is lighter than training, so 50-100 images usually fit.
    with torch.no_grad():
        # returns bag_logits: [1, Classes], inst_logits: [1, N, Classes], attn: [1, N, 1]
        bag_logits, inst_logits, attn = model(img_tensor)
        
    # 3. Process Bag Prediction
    bag_probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(bag_probs, dim=0)
    bag_class = class_list[bag_pred_idx.item()]
    
    # 4. Process Foreign Objects
    foreign_objects = []
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0) # [N, Classes]
    p_confs, p_preds = torch.max(inst_probs, dim=1)
    
    for i in range(len(p_confs)):
        p_idx = p_preds[i].item()
        p_conf = p_confs[i].item()
        
        # LOGIC: Flag if Pill != Bag AND Confidence > Threshold
        if p_idx != bag_pred_idx.item():
            if p_conf > FOREIGN_THRESH:
                foreign_objects.append({
                    "file": os.path.basename(valid_paths[i]),
                    "predicted_as": class_list[p_idx],
                    "confidence": p_conf
                })

    return {
        "bag_pred": bag_class,
        "bag_conf": bag_conf.item(),
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Folder of images or folder of bags")
    parser.add_argument("--model_path", type=str, required=True, help="Path to mil_end_to_end_final_ddp.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model, class_list = load_model(args.model_path, device)
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detect Tasks
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    
    if direct_jpgs:
        print(f"--> Mode: Single Bag ({len(direct_jpgs)} images)")
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        print(f"--> Mode: Batch Processing")
        subdirs = sorted([os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))
            
    print("\n" + "="*100)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'CONF':<8} | {'STATUS':<15}")
    print("-" * 100)
    
    for bag_id, img_paths in tasks:
        res = predict_bag(img_paths, model, device, class_list, tfm)
        
        if res:
            fo_count = len(res['foreign_objects'])
            
            status = "✅ OK"
            if fo_count > 0:
                # Simple heuristic: If >5% of pills disagree, flag it.
                if (fo_count / len(img_paths)) > 0.05:
                    status = f"⚠️ CHECK ({fo_count})"
            
            print(f"{bag_id[:25]:<25} | {res['bag_pred'][:20]:<20} | {res['bag_conf']:.1%}   | {status:<15}")
            
            if "CHECK" in status:
                print(f"   [!] Detailed Analysis for {bag_id}:")
                for fo in res['foreign_objects']:
                    print(f"       -> {fo['file']} looks like {fo['predicted_as']} ({fo['confidence']:.1%})")
                print("-" * 100)
                
    print("="*100)

if __name__ == "__main__":
    main()
