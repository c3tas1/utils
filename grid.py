def reconstruct_with_spatial_sync(all_detections, padded_grid, row_capacities):
    """
    padded_grid: The raw list_1 with '00000' and 'sectionbreak'
    all_detections: Detected SKUs with (row, slot)
    row_capacities: Total YOLO boxes per row
    """
    # 1. Map detections for easy access
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}
    
    # 2. Find the "Anchor Row"
    # We need a row that has at least one OCR hit to find the 'Global Shift'
    anchor_row_idx = -1
    local_to_padded_offset = 0
    
    for row_idx, capacity in row_capacities.items():
        row_dets = {d['slot']: d['sku'] for d in all_detections if d['row'] == row_idx}
        if not row_dets: continue
        
        # Match a detected SKU to its index in the PADDED grid
        for local_slot, sku in row_dets.items():
            if sku in padded_grid[row_idx]:
                padded_idx = padded_grid[row_idx].index(sku)
                # This offset tells us: "Slot 0 in YOLO is Index X in the padded grid"
                local_to_padded_offset = padded_idx - local_slot
                anchor_row_idx = row_idx
                break
        if anchor_row_idx != -1: break

    # 3. Reconstruct ALL rows using the same Spatial Offset
    reconstructed_shelf = []
    for i in range(len(padded_grid)):
        num_slots = row_capacities.get(i, 0)
        current_row_output = []
        
        for j in range(num_slots):
            # Check for OCR first
            if (i, j) in detection_map:
                current_row_output.append(detection_map[(i, j)])
            else:
                # Apply the spatial offset to the PADDED grid
                target_padded_idx = local_to_padded_offset + j
                
                if 0 <= target_padded_idx < len(padded_grid[i]):
                    val = padded_grid[i][target_padded_idx]
                    
                    # If we hit a '00000' or 'sectionbreak', we have a 'Dilation' issue.
                    # We must "skip" the noise to find the next valid SKU for that slot.
                    if val in ("00000", "sectionbreak"):
                        # Look forward in the padded grid until we find a real SKU
                        # this ensures we don't put a '00000' string into your OSA report
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