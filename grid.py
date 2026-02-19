def reconstruct_verified_shelf(all_detections, padded_grid, row_capacities):
    # 1. Clean the Detections and find valid anchors
    # We use a list to store all possible valid offsets found in the image
    valid_offsets = []
    
    for d in all_detections:
        row_idx = d['row']
        local_slot = d['slot']
        # Clean the string to ensure no whitespace/casing issues
        detected_sku = str(d['sku']).strip().upper()
        
        if row_idx >= len(padded_grid):
            continue
            
        # Get the target planogram row
        p_row = [str(x).strip().upper() for x in padded_grid[row_idx]]
        
        # Check if detected SKU exists in this row
        if detected_sku in p_row:
            # Find ALL occurrences (in case of multiples)
            indices = [i for i, x in enumerate(p_row) if x == detected_sku]
            for p_idx in indices:
                # Calculate the offset for this specific match
                offset = p_idx - local_slot
                valid_offsets.append(offset)

    if not valid_offsets:
        return "ERROR: No valid anchors found in the planogram."

    # 2. Use the "Consensus" Offset (The most frequent one)
    # This prevents one bad OCR/Duplicate SKU from ruining the alignment
    from collections import Counter
    global_offset = Counter(valid_offsets).most_common(1)[0][0]

    # 3. RECONSTRUCTION
    reconstructed_shelf = []
    
    for i, full_padded_row in enumerate(padded_grid):
        num_slots = row_capacities.get(i, 0)
        row_output = []
        
        # Mapping detections for THIS row only
        row_dets = {d['slot']: str(d['sku']).strip().upper() for d in all_detections if d['row'] == i}
        
        # Clean the planogram row for this iteration
        clean_p_row = [str(x).strip().upper() for x in full_padded_row]

        for j in range(num_slots):
            # Calculate where this YOLO slot maps to in the Padded Planogram
            target_p_idx = global_offset + j
            
            # Use the OCR result ONLY if it matches the planogram at that specific coordinate
            # This is the 'Verification Gate'
            detected_at_slot = row_dets.get(j)
            
            if (0 <= target_p_idx < len(clean_p_row) and 
                detected_at_slot == clean_p_row[target_p_idx]):
                row_output.append(detected_at_slot)
            else:
                # Fill from Planogram Truth
                if 0 <= target_p_idx < len(clean_p_row):
                    val = clean_p_row[target_p_idx]
                    
                    # Logic to skip 00000/sectionbreak to find the actual SKU
                    if val in ("00000", "SECTIONBREAK", ""):
                        search_idx = target_p_idx
                        while (search_idx < len(clean_p_row) and 
                               clean_p_row[search_idx] in ("00000", "SECTIONBREAK", "")):
                            search_idx += 1
                        
                        if search_idx < len(clean_p_row):
                            row_output.append(clean_p_row[search_idx])
                        else:
                            row_output.append("EMPTY")
                    else:
                        row_output.append(val)
                else:
                    row_output.append("OOB")
                    
        reconstructed_shelf.append(row_output)
        
    return reconstructed_shelf