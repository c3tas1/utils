def get_top_vertical_matches(target_row_idx, list1, list2, top_n=3):
    # 1. Identify all boxes in the row we want to fill
    target_row_boxes = list2[target_row_idx]
    
    # 2. Collect all valid candidates from other rows that HAVE labels
    candidates = []
    for r_idx, row_boxes in enumerate(list2):
        if r_idx == target_row_idx or not list1[r_idx]:
            continue
        for b_idx, box in enumerate(row_boxes):
            if b_idx < len(list1[r_idx]): # Ensure there is a corresponding label
                candidates.append({
                    'coords': box,
                    'row': r_idx,
                    'col': b_idx,
                    'val': list1[r_idx][b_idx]
                })

    all_potential_matches = []

    # 3. For every box in our target row, find how well it matches candidates
    for t_idx, t_box in enumerate(target_row_boxes):
        t_x = (t_box[0] + t_box[2]) / 2
        t_y = (t_box[1] + t_box[3]) / 2

        for cand in candidates:
            c_box = cand['coords']
            c_x = (c_box[0] + c_box[2]) / 2
            c_y = (c_box[1] + c_box[3]) / 2

            dx = abs(t_x - c_x)
            dy = abs(t_y - c_y)

            # Scoring: Vertical distance + Heavy Horizontal Penalty
            # This ensures we favor boxes in the same column
            score = math.sqrt((dx * 25)**2 + dy**2)

            all_potential_matches.append({
                'target_box_idx': t_idx,
                'matched_val': cand['val'],
                'matched_row': cand['row'],
                'matched_col': cand['col'],
                'score': score
            })

    # 4. Sort by best score and return top N
    all_potential_matches.sort(key=lambda x: x['score'])
    return all_potential_matches[:top_n]