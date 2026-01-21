import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import sys

# --- CONFIG ---
BACKBONE_BATCH_SIZE = 256
# Standard Cosine Similarity Threshold
# 1.0 = Perfect Match. 0.0 = Orthogonal.
# Real pills usually score > 0.7. Foreign objects usually score < 0.4.
SIMILARITY_THRESH = 0.5 

# --- MIL MODEL DEFINITION (To identify the bag class) ---
class MILHead(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(input_dim, num_classes)
        self.instance_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1) 
        bag_embedding = torch.sum(x * A, dim=1) 
        return self.bag_classifier(bag_embedding), None, A

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone...")
    model = models.resnet34(weights=None)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if 'fc.weight' in new_state_dict:
        model.fc = nn.Linear(512, new_state_dict['fc.weight'].shape[0])
    model.load_state_dict(new_state_dict, strict=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def predict_with_prototype(img_paths, backbone, mil_model, prototypes, device, class_list, transform):
    if not img_paths: return None

    # 1. Extract Features
    features_list = []
    for i in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
        batch_paths = img_paths[i : i+BACKBONE_BATCH_SIZE]
        images = []
        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    images.append(transform(img.convert('RGB')))
            except: pass
        if not images: continue
        
        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats = backbone(img_tensor)
            # Normalize immediately for cosine similarity
            feats = F.normalize(feats.float(), p=2, dim=1) 
            features_list.append(feats)
            
    if not features_list: return None
    
    # [N, 512] - Individual Pill Vectors
    all_feats = torch.cat(features_list, dim=0)
    # [1, N, 512] - For MIL
    bag_input = all_feats.unsqueeze(0)

    # 2. Identify the Bag Class
    with torch.no_grad():
        bag_logits, _, _ = mil_model(bag_input)
        
    probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(probs, dim=0)
    predicted_class = class_list[bag_pred_idx.item()]
    
    # 3. Load the Prototype (The Golden Vector)
    if predicted_class not in prototypes:
        return {"error": f"No prototype found for {predicted_class}"}
        
    # [1, 512]
    golden_vector = prototypes[predicted_class].to(device).unsqueeze(0)
    
    # 4. Compare Every Pill to the Golden Vector
    # Shape: [N, 1]
    similarities = torch.mm(all_feats, golden_vector.t()).squeeze(1)
    
    foreign_objects = []
    
    for i in range(len(img_paths)):
        score = similarities[i].item()
        
        if score < SIMILARITY_THRESH:
            foreign_objects.append({
                "file": os.path.basename(img_paths[i]),
                "score": score
            })
            
    return {
        "class": predicted_class,
        "conf": bag_conf.item(),
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--backbone_path", type=str, required=True)
    parser.add_argument("--mil_path", type=str, required=True)
    parser.add_argument("--prototypes_path", type=str, default="class_prototypes.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    mil_chk = torch.load(args.mil_path, map_location=device)
    class_list = mil_chk['classes']
    mil_model = MILHead(len(class_list)).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'])
    mil_model.eval()
    
    backbone = load_backbone(args.backbone_path, device)
    
    # Load Prototypes
    print(f"--> Loading Prototypes from {args.prototypes_path}")
    prototypes = torch.load(args.prototypes_path)
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detect Input
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    if direct_jpgs:
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        subdirs = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))

    print("\n" + "="*80)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'ANOMALIES':<10}")
    print("-" * 80)
    
    for bag_id, img_paths in tasks:
        res = predict_with_prototype(img_paths, backbone, mil_model, prototypes, device, class_list, tfm)
        
        if "error" in res:
            print(f"{bag_id:<25} | ERROR: {res['error']}")
            continue
            
        fo_count = len(res['foreign_objects'])
        status = f"YES ({fo_count})" if fo_count > 0 else "NO"
        
        print(f"{bag_id[:25]:<25} | {res['class'][:20]:<20} | {status:<10}")
        
        if res['foreign_objects']:
            res['foreign_objects'].sort(key=lambda x: x['score'])
            for fo in res['foreign_objects']:
                print(f"   -> {fo['file']} (Similarity: {fo['score']:.4f})")
            print("-" * 80)
            
    print("="*80)

if __name__ == "__main__":
    main()
