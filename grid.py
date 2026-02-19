def get_center(bbox):
    # Returns [x_center, y_center]
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def find_top_3_near_row2(list1, list2, target_row=2):
    all_matches = []
    
    # 1. These are the boxes we are trying to find neighbors FOR
    empty_row_boxes = list2[target_row]
    
    # 2. Loop through every box in the empty row (the "Green Circle" ones)
    for empty_box_idx, e_box in enumerate(empty_row_boxes):
        e_center = get_center(e_box)
        
        # 3. Loop through every other row in the dataset
        for r_idx, row_boxes in enumerate(list2):
            # We only check rows that ARE NOT the empty one AND have values in list1
            if r_idx == target_row or len(list1[r_idx]) == 0:
                continue
            
            # 4. Loop through every box in these valid rows
            for c_idx, v_box in enumerate(row_boxes):
                v_center = get_center(v_box)
                
                # Calculate Horizontal (dx) and Vertical (dy) distance
                dx = abs(e_center[0] - v_center[0])
                dy = abs(e_center[1] - v_center[1])
                
                # We give Horizontal distance a heavy weight (20) 
                # so the nearest "Vertical" box wins even if it's further away than a side box
                score = math.sqrt((dx * 20)**2 + dy**2)
                
                all_matches.append({
                    'target_box_in_row2': empty_box_idx,
                    'match_location': (r_idx, c_idx),
                    'value': list1[r_idx][c_idx] if c_idx < len(list1[r_idx]) else "N/A",
                    'score': score
                })

    # Sort all findings by the best (lowest) score
    all_matches.sort(key=lambda x: x['score'])
    
    # Return just the top 3
    return all_matches[:3]