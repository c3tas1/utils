def reconstruct_full_shelf(all_detections, planogram_grid, row_capacities):
    """
    all_detections: List of dicts [{'row': i, 'slot': j, 'sku': '123'}]
    planogram_grid: The FULL padded list (List 1) with 00000s and sectionbreaks
    row_capacities: Dict {row_index: num_slots_yolo_saw} (e.g., {0: 6, 1: 6})
    """
    # 1. Group detections by row for easier access
    rows_with_data = {}
    for d in all_detections:
        rows_with_data.setdefault(d['row'], {})[d['slot']] = d['sku']

    # 2. Find the Global Offset (The "Master Anchor")
    # We find where the relative 'slot 0' of our detections sits in the 'full' planogram
    global_offset = None
    
    for row_idx, detected_slots in rows_with_data.items():
        # Get the clean sequence for this planogram row (ignoring 00000/sectionbreak)
        clean_p_row = [sku for sku in planogram_grid[row_idx] if sku not in ("00000", "sectionbreak")]
        
        for local_slot, sku in detected_slots.items():
            if sku in clean_p_row:
                p_index = clean_p_row.index(sku)
                # The start of the detected window in the clean planogram
                global_offset = p_index - local_slot
                break
        if global_offset is not None: break

    if global_offset is None:
        return "Error: No anchors found in any row to align planogram."

    # 3. Reconstruct every row using the inherited offset
    reconstructed_shelf = []
    for i, full_padded_row in enumerate(planogram_grid):
        # Filter out padding to work with the real product sequence
        clean_p_row = [sku for sku in full_padded_row if sku not in ("00000", "sectionbreak")]
        num_to_extract = row_capacities.get(i, 0)
        
        row_reconstruction = []
        current_row_detections = rows_with_data.get(i, {})

        for j in range(num_to_extract):
            # Check if we have an OCR result for this slot
            if j in current_row_detections:
                row_reconstruction.append(current_row_detections[j])
            else:
                # Use the inherited offset to pull from the planogram
                p_target_idx = global_offset + j
                if 0 <= p_target_idx < len(clean_p_row):
                    row_reconstruction.append(clean_p_row[p_target_idx])
                else:
                    row_reconstruction.append("VOID") # Outside planogram range
        
        reconstructed_shelf.append(row_reconstruction)

    return reconstructed_shelf

# --- Test Scenario ---
# 100-item planogram, but camera sees only 3 slots
# Row 0: Fully OCR'd. Row 1: Missing (Zero detections). Row 2: Partial OCR.
planogram_grid = [
    ["00000", "A1", "A2", "A3", "A4", "A5"], # Assume these are 100 items long
    ["00000", "B1", "B2", "B3", "B4", "B5"],
    ["00000", "C1", "C2", "C3", "C4", "C5"]
]
# YOLO saw 3 slots on every shelf
row_caps = {0: 3, 1: 3, 2: 3} 

# Detections: Note that Row 1 is totally missing
detections = [
    {'row': 0, 'slot': 0, 'sku': "A1"}, {'row': 0, 'slot': 1, 'sku': "A2"}, {'row': 0, 'slot': 2, 'sku': "A3"},
    {'row': 2, 'slot': 1, 'sku': "C2"} # Only 1 anchor in Row 2
]

final_shelf = reconstruct_full_shelf(detections, planogram_grid, row_caps)