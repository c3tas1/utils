import cv2
import numpy as np
import os

def visualize_sku_matches(reference_path, shelf_image_path, output_name):
    # 1. Load images in Grayscale for matching, and Color for visualization
    template = cv2.imread(reference_path)
    shelf = cv2.imread(shelf_image_path)
    
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_shelf = cv2.cvtColor(shelf, cv2.COLOR_BGR2GRAY)

    # 2. Initialize SIFT (Highly robust for retail environments)
    sift = cv2.SIFT_create(nfeatures=2000) # Increased features for high-res shelves

    # 3. Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray_template, None)
    kp2, des2 = sift.detectAndCompute(gray_shelf, None)

    # 4. FLANN Matcher configuration
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5. KNN Match
    matches = flann.knnMatch(des1, des2, k=2)

    # 6. Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 7. Draw the matches
    if len(good_matches) > 5:
        # drawMatches draws lines between the two images
        match_viz = cv2.drawMatches(
            template, kp1, 
            shelf, kp2, 
            good_matches, None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0), # Green lines for matches
            singlePointColor=(0, 0, 255)
        )
        
        # 8. Optional: Draw a bounding box around the detection on the shelf side
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w, _ = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Note: We must offset the bounding box by the width of the template 
            # because drawMatches places the shelf image to the right of the template
            dst += (w, 0) 
            cv2.polylines(match_viz, [np.int32(dst)], True, (255, 0, 0), 5)

        cv2.imwrite(output_name, match_viz)
        print(f"Success: {output_name} saved with {len(good_matches)} matches.")
    else:
        print(f"Insufficient matches for {reference_path}.")

# Execution
skus_dir = "./skulabels/"
shelf_img = "shelf_full.jpg"

for sku_file in os.listdir(skus_dir):
    if sku_file.endswith(('.jpg', '.png', '.jpeg')):
        full_path = os.path.join(skus_dir, sku_file)
        visualize_sku_matches(full_path, shelf_img, f"match_result_{sku_file}")