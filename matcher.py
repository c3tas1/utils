import cv2
import numpy as np
import os

def match_sku_to_shelf(reference_path, shelf_image_path):
    # 1. Load images
    template = cv2.imread(reference_path, 0)  # Reference SKU crop
    shelf = cv2.imread(shelf_image_path, 0)   # Full shelf image
    shelf_color = cv2.imread(shelf_image_path)

    # 2. Initialize SIFT detector
    sift = cv2.SIFT_create()

    # 3. Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(shelf, None)

    # 4. FLANN parameters for fast matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5. Perform Matching using K-Nearest Neighbors
    matches = flann.knnMatch(des1, des2, k=2)

    # 6. Lowe's Ratio Test to filter weak matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 7. Localize the SKU if enough matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography (the perspective mapping)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Draw bounding box on shelf image
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        result_img = cv2.polylines(shelf_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        return result_img, len(good_matches)
    
    return None, 0

# Usage Loop
shelf_img = "shelf_image.jpg"
skus_folder = "./skulabels/"

for sku_file in os.listdir(skus_folder):
    result, match_count = match_sku_to_shelf(os.path.join(skus_folder, sku_file), shelf_img)
    if result is not None:
        print(f"Matched {sku_file} with {match_count} points.")
        cv2.imwrite(f"detected_{sku_file}", result)