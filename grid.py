def reconstruct_with_verified_anchors(all_detections, padded_grid, row_capacities):
    """
    all_detections: Detected SKUs with (row, slot, sku)
    padded_grid: The raw list_1 with '00000' and 'sectionbreak'
    row_capacities: Total YOLO boxes per row
    """
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}
    
    # 1. ANCHOR SEARCH & VALIDATION
    local_to_padded_offset = None
    
    # Sort detections to prioritize rows with more hits for better anchor stability
    for d in all_detections:
        row_idx, local_slot, sku = d['row'], d['slot'], d['sku']
        
        # VALIDATION GATE: Only use this as an anchor if it exists in the planogram row
        if sku in padded_grid[row_idx]:
            padded_idx = padded_grid[row_idx].index(sku)
            local_to_padded_offset = padded_idx - local_slot
            break # Found a verified anchor, we can stop searching
        else:
            # Optional: Log the misdetection for model retraining
            print(f"Ignoring invalid detection: {sku} at row {row_idx}")

    # 2. FAIL-SAFE: If no detections match the planogram at all
    if local_to_padded_offset is None:
        return "ERROR: No valid anchors found. All OCR results mismatch planogram."

    # 3. RECONSTRUCTION
    reconstructed_shelf = []
    for i in range(len(padded_grid)):
        num_slots = row_capacities.get(i, 0)
        current_row_output = []
        
        for j in range(num_slots):
            # Check if this specific slot has a detection
            raw_sku = detection_map.get((i, j))
            
            # Even for the slot itself, we validate: 
            # If the detected SKU isn't in the planogram, we ignore it and use the planogram instead
            if raw_sku and raw_sku in padded_grid[i]:
                current_row_output.append(raw_sku)
            else:
                # Use the verified spatial offset
                target_padded_idx = local_to_padded_offset + j
                
                if 0 <= target_padded_idx < len(padded_grid[i]):
                    val = padded_grid[i][target_padded_idx]
                    
                    # Jump padding/breaks to find the nearest real SKU
                    if val in ("00000", "sectionbreak"):
                        search_idx = target_padded_idx
                        while search_idx < len(padded_grid[i]) and padded_grid[i][search_idx] in ("00000", "sectionbreak"):
                            search_idx += 1
                        
                        if search_idx < len(padded_grid[i]):
                            current_row_output.append(padded_grid[i][search_idx])
                        else:
                            current_row_output.append("EMPTY")
                    else:
                        current_row_output.append(val)
                else:
                    current_row_output.append("OUT_OF_BOUNDS")
                    
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf