def reconstruct_with_spatial_sync(all_detections, padded_grid, row_capacities):
    """
    padded_grid: The raw list_1 with '00000' and 'sectionbreak'
    all_detections: Detected SKUs with (row, slot, sku)
    row_capacities: Total YOLO boxes per row
    """
    # 1. Map detections for easy access
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}
    
    # 2. Global Anchor Search
    # We find the first detection that exists ANYWHERE in the planogram grid
    global_row_offset = None
    global_col_offset = None
    
    for det in all_detections:
        sku = det['sku']
        found = False
        
        # Search every row in the planogram, not just the detected row index
        for p_row_idx, p_row in enumerate(padded_grid):
            if sku in p_row:
                p_col_idx = p_row.index(sku)
                # Offset: Mapping YOLO (0,0) to Planogram (p_row, p_col)
                global_row_offset = p_row_idx - det['row']
                global_col_offset = p_col_idx - det['slot']
                found = True
                break
        
        if found:
            break # Anchor established, move to reconstruction
            
    # If no valid anchor is found in the whole grid, we can't align
    if global_row_offset is None:
        return []

    # 3. Reconstruction
    reconstructed_shelf = []
    # Iterate through the rows the camera actually saw
    for i in sorted(row_capacities.keys()):
        num_slots = row_capacities[i]
        current_row_output = []
        
        # Calculate which planogram row corresponds to this camera row
        target_p_row_idx = i + global_row_offset
        
        for j in range(num_slots):
            sku_at_slot = detection_map.get((i, j))
            
            # Validation: Only trust OCR if it matches the planogram at this row
            is_valid_ocr = False
            if 0 <= target_p_row_idx < len(padded_grid):
                if sku_at_slot and sku_at_slot in padded_grid[target_p_row_idx]:
                    is_valid_ocr = True

            if is_valid_ocr:
                current_row_output.append(sku_at_slot)
            else:
                # Apply the global spatial shift to the padded grid
                target_padded_idx = global_col_offset + j
                
                if 0 <= target_p_row_idx < len(padded_grid):
                    p_row = padded_grid[target_p_row_idx]
                    
                    if 0 <= target_padded_idx < len(p_row):
                        val = p_row[target_padded_idx]
                        
                        # Dilation logic: skip "00000" or "sectionbreak" to find the real SKU
                        if val in ("00000", "sectionbreak"):
                            search_idx = target_padded_idx
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
                        current_row_output.append("OOB")
                else:
                    current_row_output.append("OOB")
                    
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf