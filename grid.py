def reconstruct_shelf_final(all_detections, padded_grid, row_capacities):
    """
    all_detections: List of dicts [{'row': i, 'slot': j, 'sku': '123'}]
    padded_grid: The full 2D list (List 1) with '00000' and 'sectionbreak'
    row_capacities: Dict {yolo_row_idx: box_count} (e.g., {0: 4, 1: 3, 2: 4})
    """
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}
    
    # 1. Global Anchor Search
    # We find where a detected SKU sits in the WHOLE planogram to lock coordinates
    global_row_offset = None
    global_col_offset = None

    for det in all_detections:
        det_row, det_slot, sku = det['row'], det['slot'], det['sku']
        
        found_in_planogram = False
        # Search every row and column of the planogram grid
        for p_row_idx, p_row in enumerate(padded_grid):
            if sku in p_row:
                p_col_idx = p_row.index(sku)
                
                # Calculate offsets: "How many rows/cols is the camera shifted from index 0,0?"
                global_row_offset = p_row_idx - det_row
                global_col_offset = p_col_idx - det_slot
                found_in_planogram = True
                break
        
        if found_in_planogram:
            break
        else:
            # REQ: If detected SKU is not in the planogram grid at all, raise error
            raise ValueError(f"Detection Error: SKU '{sku}' not found in the planogram grid.")

    if global_row_offset is None:
        raise ValueError("Mapping Error: No valid anchors could be established.")

    # 2. Reconstruction using the Spatial Offsets
    reconstructed_shelf = []
    
    # We iterate based on the number of rows the YOLO model actually saw
    for i in sorted(row_capacities.keys()):
        num_slots = row_capacities[i]
        current_row_output = []
        
        # Map the current camera row to the correct planogram row
        target_p_row_idx = i + global_row_offset
        
        for j in range(num_slots):
            # Check for OCR hits on this specific shelf/slot first
            if (i, j) in detection_map:
                current_row_output.append(detection_map[(i, j)])
            else:
                # Apply the global spatial shift to the padded planogram
                target_p_col_idx = global_col_offset + j
                
                if 0 <= target_p_row_idx < len(padded_grid):
                    p_row = padded_grid[target_p_row_idx]
                    
                    if 0 <= target_p_col_idx < len(p_row):
                        val = p_row[target_p_col_idx]
                        
                        # Logic to skip padding/breaks to find the next valid SKU
                        if val in ("00000", "sectionbreak"):
                            search_idx = target_p_col_idx
                            while (search_idx < len(p_row) and 
                                   p_row[search_idx] in ("00000", "sectionbreak")):
                                search_idx += 1
                            
                            if search_idx < len(p_row):
                                current_row_output.append(p_row[search_idx])
                            else:
                                current_row_output.append("EMPTY")
                        else:
                            current_row_output.append(val)
                    else:
                        current_row_output.append("COL_OUT_OF_BOUNDS")
                else:
                    current_row_output.append("ROW_OUT_OF_BOUNDS")
                    
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf