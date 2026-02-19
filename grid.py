def get_clean_sequence(padded_row):
    """Returns a list of (original_index, sku) for real products only."""
    return [(i, sku) for i, sku in enumerate(padded_row) 
            if sku not in ("00000", "sectionbreak", "")]

def generate_reality_for_anchor(anchor_sku, local_idx, num_slots, padded_row):
    # 1. Get the real sequence of products for this shelf
    clean_seq = get_clean_sequence(padded_row)
    
    # 2. Find where our anchor SKU sits in the physical product list
    # We find the matching SKU that is most likely to be at this relative position
    match_idx = -1
    for i, (orig_idx, sku) in enumerate(clean_seq):
        if sku == anchor_sku:
            match_idx = i
            break
            
    if match_idx == -1: return None # SKU not in this planogram row

    # 3. Calculate where the 'Window Start' (Slot 0) sits in the Clean Sequence
    # If anchor is at slot 2 (local_idx) and is the 10th product (match_idx),
    # then our window starts at the 8th physical product.
    start_clean_idx = match_idx - local_idx
    
    reality = []
    for k in range(num_slots):
        target_clean_idx = start_clean_idx + k
        
        if 0 <= target_clean_idx < len(clean_seq):
            # Fetch the actual SKU from the physical sequence
            reality.append(clean_seq[target_clean_idx][1])
        else:
            reality.append("VOID")
            
    return reality

def resolve_row_truth(row_detections, num_slots, padded_row):
    realities = []
    
    # Create a reality for every single SKU detection in this row
    for slot_idx, sku in row_detections.items():
        r = generate_reality_for_anchor(sku, slot_idx, num_slots, padded_row)
        if r: realities.append(r)
        
    if not realities: return None
    
    # Scoring: Pick the reality that matches the most OCR results
    # (Your existing 'get_common_reality' logic)
    best_reality = max(realities, key=lambda r: sum(1 for idx, s in enumerate(r) 
                                                   if idx in row_detections and s == row_detections[idx]))
    return best_reality