def reconstruct_shelf_grid(all_detections, planogram_grid, row_capacities):
    """
    all_detections: [{'row': i, 'slot': j, 'sku': '123'}]
    planogram_grid: The raw nested list with '00000' and 'sectionbreak'
    row_capacities: {row_idx: total_yolo_boxes_seen}
    """
    
    # 1. Find the FIRST valid anchor to establish the global spatial shift
    # We need to know: Slot 'j' in YOLO corresponds to Padded Index 'k' in Planogram
    global_shift = None
    
    for det in all_detections:
        r_idx, local_slot, sku = det['row'], det['slot'], det['sku']
        
        # Validation: Is this SKU actually in the planogram row?
        if sku in planogram_grid[r_idx]:
            # Find the physical index in the padded grid
            padded_index = planogram_grid[r_idx].index(sku)
            # Calculation: If Slot 1 is Padded Index 10, the window starts at Padded Index 9
            global_shift = padded_index - local_slot
            break
        else:
            # If the user's detected SKU isn't in the planogram, it's a hard fail/ignore
            raise ValueError(f"Detection Error: SKU '{sku}' not found in planogram row {r_idx}")

    if global_shift is None:
        raise ValueError("Mapping Error: No valid anchors could be established from detections.")

    # 2. Reconstruct the shelf using the established shift
    reconstructed_shelf = []
    
    for i in range(len(planogram_grid)):
        num_items_to_fetch = row_capacities.get(i, 0)
        current_row_output = []
        
        # Local map for OCR hits in this specific row
        row_dets = {d['slot']: d['sku'] for d in all_detections if d['row'] == i}
        
        for j in range(num_items_to_fetch):
            # If we have an OCR hit for this slot, use it
            if j in row_dets:
                current_row_output.append(row_dets[j])
            else:
                # Calculate the target index in the padded planogram
                target_idx = global_shift + j
                
                if 0 <= target_idx < len(planogram_grid[i]):
                    val = planogram_grid[i][target_idx]
                    
                    # If we hit padding or a break, look for the next real SKU
                    if val in ("00000", "sectionbreak"):
                        search_ptr = target_idx
                        # Move forward until a real SKU is found
                        while (search_ptr < len(planogram_grid[i]) and 
                               planogram_grid[i][search_ptr] in ("00000", "sectionbreak")):
                            search_ptr += 1
                        
                        if search_ptr < len(planogram_grid[i]):
                            current_row_output.append(planogram_grid[i][search_ptr])
                        else:
                            current_row_output.append("EMPTY")
                    else:
                        current_row_output.append(val)
                else:
                    current_row_output.append("OUT_OF_BOUNDS")
        
        reconstructed_shelf.append(current_row_output)
        
    return reconstructed_shelf