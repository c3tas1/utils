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
        t_cy = (t_y1 + t_y2) / 2
        t_width = t_x2 - t_x1
        t_height = t_y2 - t_y1
        is_empty = targets.class_id[t_idx] == 1
        
        best_idx, min_score = -1, float('inf')
        multi_matches = []
        
        for l_idx, l_box in enumerate(labels.xyxy):
            l_x1, l_y1, l_x2, l_y2 = l_box
            l_cx = (l_x1 + l_x2) / 2
            l_cy = (l_y1 + l_y2) / 2
            l_width = l_x2 - l_x1
            l_height = l_y2 - l_y1
            
            # Calculate vertical relationship
            v_gap = l_y1 - t_y2  # Positive if label fully below target
            
            # Calculate overlap between boxes
            overlap_x = max(0, min(t_x2, l_x2) - max(t_x1, l_x1))
            overlap_y = max(0, min(t_y2, l_y2) - max(t_y1, l_y1))
            
            # Horizontal metrics
            h_dist = abs(l_cx - t_cx)
            h_overlap_ratio = overlap_x / l_width if l_width > 0 else 0
            
            # NEW: Check if label is in valid position relative to target
            # Valid positions:
            # 1. Below target (v_gap >= 0)
            # 2. Overlapping with target (overlap_y > 0) AND label center is in lower half of target or below
            # 3. Slightly above but very close (v_gap >= -small_threshold)
            
            label_below_target_center = l_cy > t_cy
            boxes_overlap = overlap_x > 0 and overlap_y > 0
            
            # Determine if this is a valid spatial relationship
            is_valid_position = False
            
            if v_gap >= 0 and v_gap <= MAX_V_GAP:
                # Case 1: Label is cleanly below target
                is_valid_position = True
                position_type = "below"
                
            elif boxes_overlap and label_below_target_center:
                # Case 2: Boxes overlap but label center is in lower portion
                is_valid_position = True
                position_type = "overlapping"
                
            elif v_gap >= -l_height * 0.5 and v_gap < 0:
                # Case 3: Label slightly overlaps from below (partial overlap)
                # Only valid if there's good horizontal alignment
                if h_overlap_ratio > 0.3 or h_dist < t_width * 0.5:
                    is_valid_position = True
                    position_type = "partial_overlap"
            
            if not is_valid_position:
                continue
            
            # Check horizontal alignment
            # Label should be roughly under/aligned with target
            if h_dist > t_width * 0.75 and h_overlap_ratio < 0.15:
                continue
            
            # Calculate association score (lower = better)
            if is_empty:
                # Empty slots can span multiple labels
                if h_overlap_ratio > 0.2 or h_dist < t_width * 0.5:
                    multi_matches.append({
                        "label_id": l_idx,
                        "label_xyxy": l_box,
                        "target_type": "Empty",
                        "target_xyxy": t_box,
                        "position_type": position_type
                    })
            else:
                # Single object to single label
                # Scoring: prioritize horizontal alignment, then vertical proximity
                
                if position_type == "overlapping":
                    # For overlapping boxes, use center-to-center distance
                    # but weight horizontal more heavily
                    score = h_dist * 1.5 + abs(l_cy - t_cy) * 0.5
                    # Bonus for good horizontal overlap
                    score -= h_overlap_ratio * 100
                    
                elif position_type == "partial_overlap":
                    # Partial overlap - moderate scoring
                    score = h_dist + abs(v_gap) * 0.5
                    score -= h_overlap_ratio * 50
                    
                else:  # "below"
                    # Standard case: label below target
                    score = h_dist + v_gap * 0.3
                    score -= h_overlap_ratio * 50
                
                if score < min_score:
                    min_score = score
                    best_idx = l_idx
        
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
