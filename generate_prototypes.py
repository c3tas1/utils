import os
import torch
import glob
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    # Point this to the .pt files created by preprocess_optimized.py
    parser.add_argument("--features_dir", type=str, required=True, help="./features_data/train")
    parser.add_argument("--save_path", type=str, default="class_prototypes.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Computing Class Prototypes (Centroids) on {device}...")

    # Dictionary to store sum of features and count
    # Structure: { "Amoxicillin": {'sum': Tensor, 'count': int} }
    class_stats = {}

    classes = sorted([d for d in os.listdir(args.features_dir) if os.path.isdir(os.path.join(args.features_dir, d))])
    
    for cls in tqdm(classes):
        cls_dir = os.path.join(args.features_dir, cls)
        pt_files = glob.glob(os.path.join(cls_dir, "*.pt"))
        
        if not pt_files: continue
        
        feature_sum = None
        count = 0
        
        for p in pt_files:
            # Load [N_pills, 512]
            feats = torch.load(p, map_location=device)
            
            # Normalize vectors first! 
            # (Crucial: we want the average DIRECTION, not average magnitude)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            if feature_sum is None:
                feature_sum = torch.sum(feats, dim=0)
            else:
                feature_sum += torch.sum(feats, dim=0)
                
            count += feats.shape[0]
            
        if count > 0:
            # Calculate Mean Vector (Centroid)
            centroid = feature_sum / count
            # Re-normalize the centroid so it lies on the unit sphere
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
            
            class_stats[cls] = centroid.cpu()

    print(f"--> Calculated prototypes for {len(class_stats)} classes.")
    torch.save(class_stats, args.save_path)
    print(f"--> Saved to {args.save_path}")

if __name__ == "__main__":
    main()
