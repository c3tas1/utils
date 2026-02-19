def reconstruct_shelf_osa(all_detections, padded_grid, row_capacities):
    """
    all_detections: list of {'row': i, 'slot': j, 'sku': '123'}
    padded_grid: Master SKU grid (List 1) with '00000' and 'sectionbreak'
    row_capacities: dict {row_index: total_yolo_boxes}
    """
    # 1. Coordinate Locking (The "Global Anchor")
    # We find the first valid SKU to align the camera window to the planogram
    global_col_offset = None
    global_row_offset = None

    for det in all_detections:
        sku = str(det['sku']).strip()
        # Search entire grid for the anchor to handle row/col shifts
        for p_row_idx, p_row in enumerate(padded_grid):
            if sku in p_row:
                p_col_idx = p_row.index(sku)
                global_col_offset = p_col_idx - det['slot']
                global_row_offset = p_row_idx - det['row']
                break
        if global_col_offset is not None: break

    # Fallback: If no anchors match the planogram, we cannot align
    if global_col_offset is None: return []

    # 2. Reconstruct visible window for all rows
    reconstructed_shelf = []
    detection_map = {(d['row'], d['slot']): d['sku'] for d in all_detections}

    for i in sorted(row_capacities.keys()):
        num_slots = row_capacities[i]
        current_row_output = []
        # Calculate which planogram row corresponds to this physical shelf
        target_p_row_idx = i + global_row_offset
        
        for j in range(num_slots):
            # Check Reality: If OCR exists and is in the correct planogram row, trust it
            ocr_sku = detection_map.get((i, j))
            
            is_valid_ocr = False
            if 0 <= target_p_row_idx < len(padded_grid):
                if ocr_sku and ocr_sku in padded_grid[target_p_row_idx]:
                    is_valid_ocr = True

            if is_valid_ocr:
                current_row_output.append(ocr_sku)
            else:
                # Extraction logic for missing/invalid SKUs
                target_p_col_idx = global_col_offset + j
                
                if 0 <= target_p_row_idx < len(padded_grid):
                    p_row = padded_grid[target_p_row_idx]
                    
                    if 0 <= target_p_col_idx < len(p_row):
                        val = p_row[target_p_col_idx]
                        
                        # Section Jump Logic: Skip '00000' and 'sectionbreak' 
                        # to find the next valid item in the sequence
                        if val in ("00000", "sectionbreak"):
                            search_ptr = target_p_col_idx
                            while (search_ptr < len(p_row) and 
                                   p_row[search_ptr] in ("00000", "sectionbreak")):
                                search_ptr += 1
                            
                            current_row_output.append(p_row[search_ptr] if search_ptr < len(p_row) else "EMPTY")
                        else:
                            current_row_output.append(val)
                    else:
                        current_row_output.append("OOB")
        
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf