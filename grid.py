valid_row_indices = [i for i, row in enumerate(list1) if len(row) > 0]

def get_centroid(bbox):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def find_nearest_vertical_robust(empty_row_idx):
    final_results = []
    
    # Iterate through each box in the row that needs filling
    for box_idx, empty_bbox in enumerate(list2[empty_row_idx]):
        target_center = get_centroid(empty_bbox)
        candidate_matches = []
        
        for v_row_idx in valid_row_indices:
            for v_box_idx, valid_bbox in enumerate(list2[v_row_idx]):
                valid_center = get_centroid(valid_bbox)
                
                # Calculate absolute differences
                dx = abs(target_center[0] - valid_center[0])
                dy = abs(target_center[1] - valid_center[1])
                
                # WEIGHTED DISTANCE FORMULA
                # We multiply dx by a factor (e.g., 2) to penalize horizontal drift
                # but keep the logic functional even with zero overlap.
                combined_score = math.sqrt((dx * 2)**2 + dy**2)
                
                candidate_matches.append({
                    'row_idx': v_row_idx,
                    'item_idx': v_box_idx,
                    'score': combined_score,
                    'value': list1[v_row_idx][v_box_idx] if v_box_idx < len(list1[v_row_idx]) else None
                })
        
        # Find the absolute best neighbor for this specific bounding box
        if candidate_matches:
            best = min(candidate_matches, key=lambda x: x['score'])
            final_results.append(best)
            
    return final_results

# Execute
results = find_nearest_vertical_robust(2)