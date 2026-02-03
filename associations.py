def associate_mist_shelf_labels(detections, img_height):
    objs = detections[detections.class_id == 0]
    empties = detections[detections.class_id == 1]
    labels = detections[detections.class_id == 2]
    targets = sv.Detections.merge([objs, empties])
    
    associations = []
    MAX_V_GAP = img_height * 0.15
    
    # --- CHANGED: First, find the best target for EACH label ---
    # This prevents multiple targets from fighting over labels incorrectly
    label_to_target = {}  # label_idx -> (best_target_idx, score)
    
    for l_idx, l_box in enumerate(labels.xyxy):
        l_x1, l_y1, l_x2, l_y2 = l_box
        l_cx = (l_x1 + l_x2) / 2
        l_cy = (l_y1 + l_y2) / 2
        l_width = l_x2 - l_x1
        l_height = l_y2 - l_y1
        
        best_t_idx = -1
        best_score = float('inf')
        
        for t_idx, t_box in enumerate(targets.xyxy):
            t_x1, t_y1, t_x2, t_y2 = t_box
            t_cx = (t_x1 + t_x2) / 2
            t_cy = (t_y1 + t_y2) / 2
            t_width = t_x2 - t_x1
            
            v_gap = l_y1 - t_y2
            
            overlap_x = max(0, min(t_x2, l_x2) - max(t_x1, l_x1))
            overlap_y = max(0, min(t_y2, l_y2) - max(t_y1, l_y1))
            
            h_dist = abs(l_cx - t_cx)
            h_overlap_ratio = overlap_x / l_width if l_width > 0 else 0
            
            label_below_target_center = l_cy > t_cy
            boxes_overlap = overlap_x > 0 and overlap_y > 0
            label_center_within = t_x1 <= l_cx <= t_x2
            
            # Position validation
            is_valid = False
            
            if v_gap >= 0 and v_gap <= MAX_V_GAP:
                is_valid = True
            elif boxes_overlap and label_below_target_center:
                v_overlap_ratio = overlap_y / l_height if l_height > 0 else 0
                if v_overlap_ratio < 0.2 and l_cx > t_cx:
                    continue
                is_valid = True
            elif v_gap >= -l_height * 0.5 and v_gap < 0:
                if h_overlap_ratio > 0.3 or h_dist < t_width * 0.5:
                    is_valid = True
            
            if not is_valid:
                continue
            
            if h_dist > t_width * 0.75 and h_overlap_ratio < 0.15:
                continue
            
            # --- CHANGED: Simplified scoring ---
            # Priority 1: Label center is within target horizontal span (best case)
            # Priority 2: Horizontal distance (closer = better)
            # Priority 3: Small vertical gap bonus
            
            if label_center_within:
                score = h_dist  # Pure horizontal distance, will be small
            else:
                score = h_dist + t_width  # Penalty for label center outside target
            
            score += v_gap * 0.1  # Small vertical tiebreaker
            
            if score < best_score:
                best_score = score
                best_t_idx = t_idx
        
        if best_t_idx != -1:
            label_to_target[l_idx] = (best_t_idx, best_score)
    
    # --- CHANGED: Build associations from label perspective ---
    # Each label maps to exactly one target (the one most directly above it)
    target_to_labels = {}  # target_idx -> list of (label_idx, score)
    for l_idx, (t_idx, score) in label_to_target.items():
        if t_idx not in target_to_labels:
            target_to_labels[t_idx] = []
        target_to_labels[t_idx].append((l_idx, score))
    
    # For each target, pick the best label (or multiple for empties)
    for t_idx, t_box in enumerate(targets.xyxy):
        is_empty = targets.class_id[t_idx] == 1
        
        if t_idx not in target_to_labels:
            continue
        
        candidate_labels = target_to_labels[t_idx]
        
        if is_empty:
            for l_idx, score in candidate_labels:
                associations.append({
                    "label_id": l_idx,
                    "label_xyxy": labels.xyxy[l_idx],
                    "target_type": "Empty",
                    "target_xyxy": t_box
                })
        else:
            # Pick the single best label
            best_l_idx, _ = min(candidate_labels, key=lambda x: x[1])
            associations.append({
                "label_id": best_l_idx,
                "label_xyxy": labels.xyxy[best_l_idx],
                "target_type": "Object",
                "target_xyxy": t_box
            })
    
    # --- Post-process: targets with no label get left neighbor's label ---
    ROW_THRESHOLD = img_height * 0.05
    
    all_target_indices = set(range(len(targets.xyxy)))
    matched_target_indices = set(a_t_idx for a_t_idx in 
        [next((t_i for t_i in range(len(targets.xyxy)) 
               if (targets.xyxy[t_i] == a['target_xyxy']).all()), None) 
         for a in associations] if a_t_idx is not None)
    
    # Add unmatched targets with no label
    unmatched = all_target_indices - matched_target_indices
    for t_idx in unmatched:
        associations.append({
            "label_id": -1,
            "label_xyxy": None,
            "target_type": "Empty" if targets.class_id[t_idx] == 1 else "Object",
            "target_xyxy": targets.xyxy[t_idx]
        })
    
    # Group into rows and inherit from left
    sorted_assocs = sorted(associations, key=lambda a: a['target_xyxy'][1])
    
    if sorted_assocs:
        rows = []
        current_row = [sorted_assocs[0]]
        for assoc in sorted_assocs[1:]:
            curr_cy = (assoc['target_xyxy'][1] + assoc['target_xyxy'][3]) / 2
            prev_cy = (current_row[-1]['target_xyxy'][1] + current_row[-1]['target_xyxy'][3]) / 2
            if abs(curr_cy - prev_cy) < ROW_THRESHOLD:
                current_row.append(assoc)
            else:
                rows.append(current_row)
                current_row = [assoc]
        rows.append(current_row)
        
        final_associations = []
        for row in rows:
            row_sorted = sorted(row, key=lambda a: a['target_xyxy'][0])
            
            last_good_label = None
            for assoc in row_sorted:
                if assoc['label_id'] != -1:
                    last_good_label = assoc
                    final_associations.append(assoc)
                else:
                    if last_good_label is not None:
                        merged = assoc.copy()
                        merged['label_id'] = last_good_label['label_id']
                        merged['label_xyxy'] = last_good_label['label_xyxy']
                        final_associations.append(merged)
                    else:
                        final_associations.append(assoc)
        
        associations = final_associations
    
    return associations
