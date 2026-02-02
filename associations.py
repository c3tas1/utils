def associate_mist_shelf_labels(detections, img_height):
    objs = detections[detections.class_id == 0]
    empties = detections[detections.class_id == 1]
    labels = detections[detections.class_id == 2]
    targets = sv.Detections.merge([objs, empties])
    
    associations = []
    MAX_V_GAP = img_height * 0.15
    
    for t_idx, t_box in enumerate(targets.xyxy):
        t_x1, t_y1, t_x2, t_y2 = t_box
        t_cx = (t_x1 + t_x2) / 2
        t_width = t_x2 - t_x1
        is_empty = targets.class_id[t_idx] == 1
        
        best_idx, min_score = -1, float('inf')
        multi_matches = []
        
        for l_idx, l_box in enumerate(labels.xyxy):
            l_x1, l_y1, l_x2, l_y2 = l_box
            l_cx = (l_x1 + l_x2) / 2
            l_cy = (l_y1 + l_y2) / 2
            t_cy = (t_y1 + t_y2) / 2
            
            v_gap = l_y1 - t_y2  # Positive if label below target
            
            # Skip if label is above target or too far below
            if v_gap < -30 or v_gap > MAX_V_GAP:
                continue
            
            # Calculate horizontal overlap
            overlap_x = max(0, min(t_x2, l_x2) - max(t_x1, l_x1))
            l_w = l_x2 - l_x1
            overlap_ratio = overlap_x / l_w if l_w > 0 else 0
            
            # Horizontal distance: how far is label center from target center
            h_dist = abs(l_cx - t_cx)
            
            # RULE: Label should be roughly under the target
            # Allow some tolerance based on target width
            if h_dist > t_width * 0.75 and overlap_ratio < 0.1:
                continue
            
            if is_empty:
                # SPECIAL CASE: Large empty slot detection
                # More lenient - can associate with multiple labels
                if overlap_ratio > 0.2 or h_dist < t_width * 0.5:
                    multi_matches.append({
                        "label_id": l_idx,
                        "label_xyxy": l_box,
                        "target_type": "Empty",
                        "target_xyxy": t_box
                    })
            else:
                # STANDARD CASE: Single object to single SKU
                # Score = distance-based (lower is better)
                
                # Primary: prefer labels directly underneath (small h_dist)
                # Secondary: prefer closer vertical gap
                score = h_dist + (v_gap * 0.3)
                
                # Bonus for good overlap (reduce score)
                score -= (overlap_ratio * 50)
                
                if score < min_score:
                    min_score, best_idx = score, l_idx
        
        # Finalize assignment
        if is_empty and multi_matches:
            associations.extend(multi_matches)
        elif not is_empty and best_idx != -1:
            associations.append({
                "label_id": best_idx,
                "label_xyxy": labels.xyxy[best_idx],
                "target_type": "Object",
                "target_xyxy": t_box
            })
    
    return associations
