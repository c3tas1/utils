def find_top_matches_for_all_empty_rows(l1, l2, top_n=3):
    # 1. Identify which rows are empty and which are populated
    empty_row_indices = [i for i, row in enumerate(l1) if not row]
    populated_row_indices = [i for i, row in enumerate(l1) if row]

    if not empty_row_indices or not populated_row_indices:
        return "No empty rows found or no populated rows to match against."

    all_results = {}

    for target_row_idx in empty_row_indices:
        # 2. Find the nearest populated rows (Above and Below)
        # We sort populated rows by their distance to the current empty row
        nearest_rows = sorted(populated_row_indices, key=lambda i: abs(i - target_row_idx))
        
        # We only need to check the closest few populated rows to save time
        rows_to_check = nearest_rows[:4] 

        row_matches = []
        
        # 3. Get boxes for the current empty row from list2
        # Use .get() or check length to prevent index errors
        if target_row_idx >= len(l2): continue
        target_boxes = l2[target_row_idx]

        for t_box_idx, t_box in enumerate(target_boxes):
            t_center_x = (t_box[0] + t_box[2]) / 2
            t_center_y = (t_box[1] + t_box[3]) / 2

            for p_row_idx in rows_to_check:
                for p_box_idx, p_box in enumerate(l2[p_row_idx]):
                    p_center_x = (p_box[0] + p_box[2]) / 2
                    p_center_y = (p_box[1] + p_box[3]) / 2

                    # Calculate Weighted Distance
                    dx = abs(t_center_x - p_center_x)
                    dy = abs(t_center_y - p_center_y)
                    
                    # Penalty: Vertical distance is normal, 
                    # Horizontal distance (dx) is penalized 25x to keep it in column
                    score = math.sqrt((dx * 25)**2 + dy**2)

                    row_matches.append({
                        'empty_row': target_row_idx,
                        'empty_box_idx': t_box_idx,
                        'match_idx': (p_row_idx, p_box_idx),
                        'val': l1[p_row_idx][p_box_idx] if p_box_idx < len(l1[p_row_idx]) else None,
                        'score': score
                    })

        # Sort all matches for THIS empty row by score and pick top_n
        row_matches.sort(key=lambda x: x['score'])
        all_results[target_row_idx] = row_matches[:top_n]

    return all_results