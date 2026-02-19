import math

# 1. Setup your data structure
list1 = [['a', 'b'], ['d', 'e', 'f', 'g'], [], ['h', 'j', 'j', 'k', 'l']]

# Coordinates: [x_min, y_min, x_max, y_max]
list2 = [
    [[10, 10, 30, 30], [40, 10, 60, 30]],                 # Row 0
    [[10, 50, 30, 70], [40, 50, 60, 70], [70, 50, 90, 70], [100, 50, 120, 70]], # Row 1
    [[12, 100, 32, 120], [42, 100, 62, 120]],             # Row 2 (EMPTY in list1)
    [[10, 150, 30, 170], [40, 150, 60, 170], [70, 150, 90, 170], [100, 150, 120, 170], [130, 150, 150, 170]] # Row 3
]

# 2. Identify which rows actually have data in list1
valid_row_indices = [i for i, row in enumerate(list1) if len(row) > 0]

def get_centroid(bbox):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def get_x_overlap(boxA, boxB):
    # Measures how much two boxes align horizontally
    return max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]))

def find_nearest_vertical_neighbor(empty_row_idx):
    best_matches = []
    
    # Iterate through each bounding box in the empty row
    for empty_bbox in list2[empty_row_idx]:
        target_center = get_centroid(empty_bbox)
        matches_for_this_box = []
        
        for v_idx in valid_row_indices:
            for valid_bbox in list2[v_idx]:
                valid_center = get_centroid(valid_bbox)
                
                # Calculate metrics
                x_overlap = get_x_overlap(empty_bbox, valid_bbox)
                y_dist = abs(target_center[1] - valid_center[1])
                
                # Scoring: We want high x_overlap and low y_dist
                # We prioritize boxes with ANY overlap first
                score = y_dist if x_overlap > 0 else y_dist * 10 
                matches_for_this_box.append({'row': v_idx, 'score': score})
        
        # Get the best valid box for this specific empty box
        best_matches.append(min(matches_for_this_box, key=lambda x: x['score']))

    return best_matches

# 3. Execute
results = find_nearest_vertical_neighbor(2)

print(f"Results for empty Row 2:")
for i, res in enumerate(results):
    print(f"  Empty Box {i} should look at Row {res['row']} for its data.")