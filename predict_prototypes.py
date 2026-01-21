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
# Similarity Threshold using RAW Backbone Features.
# Since these are raw features, they are distinct. 
# Real pills > 0.6. Foreign Objects < 0.4.
SIMILARITY_THRESH = 0.5 

# --- TRANSFORMER MODEL (For Classification) ---
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
        return logits

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

def predict_hybrid(img_paths, backbone, mil_model, prototypes, device, class_list, transform):
    if not img_paths: return None

    # 1. Extract RAW Features (The "Truth")
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
            # Normalize Raw Features
            feats = F.normalize(feats.float(), p=2, dim=1)
            features_list.append(feats)
            
    if not features_list: return None
    
    # Raw Features [N, 512]
    raw_feats = torch.cat(features_list, dim=0)
    
    # 2. Transformer Classification (The "Brain")
    # Feed raw features into Transformer to get the Class Name
    bag_input = raw_feats.unsqueeze(0) # [1, N, 512]
    
    with torch.no_grad():
        bag_logits = mil_model(bag_input)
        
    probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(probs, dim=0)
    predicted_class = class_list[bag_pred_idx.item()]
    
    # 3. Prototype Matching (The "Filter")
    # Now we compare the RAW features against the RAW Prototype of the predicted class.
    # We DO NOT use the Transformer output here.
    
    if predicted_class not in prototypes:
        return {"error": f"Prototype missing for {predicted_class}"}
        
    # [1, 512]
    golden_vector = prototypes[predicted_class].to(device).unsqueeze(0)
    
    # Cosine Similarity: [N, 1]
    similarities = torch.mm(raw_feats, golden_vector.t()).squeeze(1)
    
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
    parser.add_argument("--input_path", type=str, required=True, help="Path to unknown prescriptions")
    parser.add_argument("--backbone_path", type=str, required=True)
    parser.add_argument("--mil_path", type=str, required=True, help="mil_transformer_ddp.pth")
    parser.add_argument("--prototypes_path", type=str, default="class_prototypes.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Transformer (For Class Name)
    print("--> Loading Transformer...")
    mil_chk = torch.load(args.mil_path, map_location=device)
    class_list = mil_chk['classes']
    mil_model = TransformerMIL(
        len(class_list), 
        hidden_dim=mil_chk.get('hidden_dim', 512),
        n_layers=mil_chk.get('n_layers', 2),
        n_heads=mil_chk.get('n_heads', 8)
    ).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'])
    mil_model.eval()
    
    # Load Backbone (For Raw Features)
    backbone = load_backbone(args.backbone_path, device)
    
    # Load Prototypes (For Outlier Detection)
    if not os.path.exists(args.prototypes_path):
        sys.exit(f"[ERROR] {args.prototypes_path} not found. Run 'generate_prototypes.py' first.")
    prototypes = torch.load(args.prototypes_path)
    print(f"--> Loaded {len(prototypes)} Class Prototypes.")
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detect Mode
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    if direct_jpgs:
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        subdirs = sorted([os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))

    print("\n" + "="*80)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'ANOMALIES':<10}")
    print("-" * 80)
    
    for bag_id, img_paths in tasks:
        res = predict_hybrid(img_paths, backbone, mil_model, prototypes, device, class_list, tfm)
        
        if "error" in res:
            print(f"{bag_id:<25} | ERROR: {res['error']}")
            continue
            
        fo_count = len(res['foreign_objects'])
        status = f"YES ({fo_count})" if fo_count > 0 else "NO"
        
        print(f"{bag_id[:25]:<25} | {res['class'][:20]:<20} | {status:<10}")
        
        if res['foreign_objects']:
            res['foreign_objects'].sort(key=lambda x: x['score'])
            for fo in res['foreign_objects'][:5]:
                print(f"   -> {fo['file']} (Score: {fo['score']:.4f})")
            if len(res['foreign_objects']) > 5: print("      ...and others")
            print("-" * 80)
            
    print("="*80)

if __name__ == "__main__":
    main()
