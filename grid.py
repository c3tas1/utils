def reconstruct_from_global_anchor(all_detections, planogram_grid, row_capacities):
    """
    planogram_grid: The full 2D padded grid (List 1)
    all_detections: Detected SKUs with relative (row, slot) from YOLO/OCR
    row_capacities: The count of YOLO boxes seen on each physical shelf
    """
    
    global_row_offset = None
    global_col_offset = None
    
    # 1. GLOBAL SEARCH: Find where a detected SKU exists ANYWHERE in the planogram
    for det in all_detections:
        detected_row, detected_slot, sku = det['row'], det['slot'], det['sku']
        
        found = False
        for p_row_idx, p_row in enumerate(planogram_grid):
            if sku in p_row:
                p_col_idx = p_row.index(sku)
                
                # Calculate the global anchor point
                # This tells us: YOLO Row 0 is actually Planogram Row X
                # and YOLO Slot 0 is actually Padded Index Y
                global_row_offset = p_row_idx - detected_row
                global_col_offset = p_col_idx - detected_slot
                found = True
                break
        
        if found: break
    
    if global_row_offset is None:
        # Instead of failing, you could iterate to the next detection 
        # but here we raise if NO detected SKUs match the master grid
        raise ValueError("Mapping Error: None of the detected SKUs exist in the planogram grid.")

    # 2. RECONSTRUCTION
    reconstructed_shelf = []
    
    for i in range(len(planogram_grid)):
        # We only reconstruct as many rows as the camera actually saw
        if i not in row_capacities: continue
        
        num_to_fetch = row_capacities[i]
        row_output = []
        
        # Local map for OCR hits on this shelf
        row_dets = {d['slot']: d['sku'] for d in all_detections if d['row'] == i}
        
        # Determine which row in the planogram_grid corresponds to this camera row
        target_p_row_idx = i + global_row_offset
        
        if 0 <= target_p_row_idx < len(planogram_grid):
            p_row = planogram_grid[target_p_row_idx]
            
            for j in range(num_to_fetch):
                if j in row_dets:
                    row_output.append(row_dets[j])
                else:
                    # Apply horizontal shift
                    target_p_col_idx = global_col_offset + j
                    
                    if 0 <= target_p_col_idx < len(p_row):
                        val = p_row[target_p_col_idx]
                        
                        # Handle padding/breaks by skipping to next SKU
                        if val in ("00000", "sectionbreak"):
                            search_ptr = target_p_col_idx
                            while (search_ptr < len(p_row) and 
                                   p_row[search_ptr] in ("00000", "sectionbreak")):
                                search_ptr += 1
                            
                            row_output.append(p_row[search_ptr] if search_ptr < len(p_row) else "EMPTY")
                        else:
                            row_output.append(val)
                    else:
                        row_output.append("OUT_OF_BOUNDS")
        
        reconstructed_shelf.append(row_output)
        
    return reconstructed_shelf