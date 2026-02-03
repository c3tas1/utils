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
            
            v_gap = l_y1 - t_y2
            
            overlap_x = max(0, min(t_x2, l_x2) - max(t_x1, l_x1))
            overlap_y = max(0, min(t_y2, l_y2) - max(t_y1, l_y1))
            
            h_dist = abs(l_cx - t_cx)
            h_overlap_ratio = overlap_x / l_width if l_width > 0 else 0
            
            label_below_target_center = l_cy > t_cy
            boxes_overlap = overlap_x > 0 and overlap_y > 0
            
            is_valid_position = False
            
            if v_gap >= 0 and v_gap <= MAX_V_GAP:
                is_valid_position = True
                position_type = "below"
                
            elif boxes_overlap and label_below_target_center:
                v_overlap_ratio = overlap_y / l_height if l_height > 0 else 0
                if v_overlap_ratio < 0.2 and l_cx > t_cx:
                    continue
                
                is_valid_position = True
                position_type = "overlapping"
                
            elif v_gap >= -l_height * 0.5 and v_gap < 0:
                if h_overlap_ratio > 0.3 or h_dist < t_width * 0.5:
                    is_valid_position = True
                    position_type = "partial_overlap"
            
            if not is_valid_position:
                continue
            
            if h_dist > t_width * 0.75 and h_overlap_ratio < 0.15:
                continue
            
            if is_empty:
                if h_overlap_ratio > 0.2 or h_dist < t_width * 0.5:
                    multi_matches.append({
                        "label_id": l_idx,
                        "label_xyxy": l_box,
                        "target_type": "Empty",
                        "target_xyxy": t_box,
                        "position_type": position_type
                    })
            else:
                if position_type == "overlapping":
                    score = h_dist * 1.5 + abs(l_cy - t_cy) * 0.5
                    score -= h_overlap_ratio * 100
                elif position_type == "partial_overlap":
                    score = h_dist + abs(v_gap) * 0.5
                    score -= h_overlap_ratio * 50
                else:
                    score = h_dist + v_gap * 0.3
                    score -= h_overlap_ratio * 50
                
                if score < min_score:
                    min_score = score
                    best_idx = l_idx
        
        if is_empty and multi_matches:
            associations.extend(multi_matches)
        elif not is_empty and best_idx != -1:
            associations.append({
                "label_id": best_idx,
                "label_xyxy": labels.xyxy[best_idx],
                "target_type": "Object",
                "target_xyxy": t_box
            })
    
    # --- Post-process: targets with no direct label underneath ---
    # inherit the nearest label from the left (same product facing)
    ROW_THRESHOLD = img_height * 0.05
    
    def has_direct_label_underneath(t_box):
        t_x1, t_y1, t_x2, t_y2 = t_box
        t_cx = (t_x1 + t_x2) / 2
        t_width = t_x2 - t_x1
        
        for l_box in labels.xyxy:
            l_x1, l_y1, l_x2, l_y2 = l_box
            l_cx = (l_x1 + l_x2) / 2
            l_width = l_x2 - l_x1
            
            v_gap = l_y1 - t_y2
            overlap_x = max(0, min(t_x2, l_x2) - max(t_x1, l_x1))
            h_overlap_ratio = overlap_x / l_width if l_width > 0 else 0
            
            label_center_within = t_x1 <= l_cx <= t_x2
            
            if 0 <= v_gap <= MAX_V_GAP and (h_overlap_ratio > 0.3 or label_center_within):
                return True
        return False
    
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
                if has_direct_label_underneath(assoc['target_xyxy']):
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
