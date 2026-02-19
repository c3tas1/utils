def reconstruct_shelf_final(all_detections, padded_grid, row_capacities):
    """
    padded_grid: Master planogram (List 1) with '00000' and 'sectionbreak'
    all_detections: OCR results with (row, slot, sku)
    row_capacities: YOLO box counts per shelf (e.g., {0: 4, 1: 3, 2: 4})
    """
    
    # 1. FIND THE GLOBAL ANCHOR
    # We search the WHOLE grid to find where the camera is looking
    global_row_offset = None
    global_col_offset = None
    
    for det in all_detections:
        sku = str(det['sku']).strip()
        for p_row_idx, p_row in enumerate(padded_grid):
            if sku in p_row:
                # Found a valid anchor in the planogram
                p_col_idx = p_row.index(sku)
                global_row_offset = p_row_idx - det['row']
                global_col_offset = p_col_idx - det['slot']
                break
        if global_row_offset is not None: break

    # If no detections match the planogram, we can't align the window
    if global_row_offset is None: return []

    # 2. RECONSTRUCT EVERY ROW
    reconstructed_shelf = []
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}

    for i in sorted(row_capacities.keys()):
        num_slots = row_capacities[i]
        current_row_output = []
        target_p_row_idx = i + global_row_offset
        
        for j in range(num_slots):
            # Check Reality (OCR) first
            sku_at_slot = detection_map.get((i, j))
            
            # Validation Gate: Ignore OCR noise if not present in this planogram row
            is_valid_ocr = False
            if 0 <= target_p_row_idx < len(padded_grid):
                if sku_at_slot and sku_at_slot in padded_grid[target_p_row_idx]:
                    is_valid_ocr = True

            if is_valid_ocr:
                current_row_output.append(sku_at_slot)
            else:
                # Use Global Offset to pull from Padded Planogram
                target_p_col_idx = global_col_offset + j
                
                if 0 <= target_p_row_idx < len(padded_grid):
                    p_row = padded_grid[target_p_row_idx]
                    
                    if 0 <= target_p_col_idx < len(p_row):
                        val = p_row[target_p_col_idx]
                        
                        # Jumping Logic (The "Smart Extract" replacement)
                        # If we hit a gap, jump to the next valid product in that section
                        if val in ("00000", "sectionbreak"):
                            search_idx = target_p_col_idx
                            while search_idx < len(p_row) and p_row[search_idx] in ("00000", "sectionbreak"):
                                search_idx += 1
                            
                            current_row_output.append(p_row[search_idx] if search_idx < len(p_row) else "EMPTY")
                        else:
                            current_row_output.append(val)
                    else:
                        current_row_output.append("OOB")
                else:
                    current_row_output.append("OOB")
                    
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf