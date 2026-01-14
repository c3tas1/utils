def group_labels_by_horizontal_rows(labels, tolerance=0):
    # If tolerance is not provided, default to 0 (strict overlap)
    # or keep your VERTICAL_TOLERANCE if you want a buffer.
    if tolerance is None:
        tolerance = 0 

    if not labels:
        return []

    label_data = []
    for l_idx, (xywh, conf, cid, l_xyxy) in labels:
        cx, cy = center_of(l_xyxy)
        label_data.append({
            'index': l_idx,
            'xywh': xywh,
            'conf': conf,
            'cid': cid,
            'xyxy': l_xyxy, # Assuming [x1, y1, x2, y2]
            'cx': cx,
            'cy': cy
        })

    # Sort by vertical position (cy) to process top-to-bottom
    label_data.sort(key=lambda x: x['cy'])

    groups = []
    current_group = [label_data[0]]

    for i in range(1, len(label_data)):
        label = label_data[i]
        
        # Get Y-bounds of the anchor (reference) and the current label
        # Assuming l_xyxy format is [x_min, y_min, x_max, y_max]
        ref_y_min = current_group[0]['xyxy'][1]
        ref_y_max = current_group[0]['xyxy'][3]
        
        curr_y_min = label['xyxy'][1]
        curr_y_max = label['xyxy'][3]

        # Calculate Vertical Intersection
        # Determine the overlap between the two vertical intervals
        # Logic: max(starts) < min(ends) implies overlap
        overlap_amount = min(ref_y_max, curr_y_max) - max(ref_y_min, curr_y_min)

        # Check for overlap
        # If overlap_amount > 0, they strictly overlap.
        # You can use 'tolerance' here to allow for negative overlap (small gaps)
        # or require a minimum positive overlap.
        if overlap_amount > tolerance:
            current_group.append(label)
        else:
            # Sort the completed group horizontally (by cx)
            current_group.sort(key=lambda x: x['cx'])
            groups.append(current_group)
            current_group = [label]

    # Append the final group
    if current_group:
        current_group.sort(key=lambda x: x['cx'])
        groups.append(current_group)

    return groups
