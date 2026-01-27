import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple

class PlanogramMatcher:
    def __init__(self, planogram_df: pd.DataFrame):
        """
        planogram_df columns: store_number, aisle_number, plano_number, 
                              shelf_number, sku_number, section_cd, plano_sequence_number
        """
        self.planogram = planogram_df
        self._build_indices()
    
    def _build_indices(self):
        """Build lookup indices for fast matching"""
        # Index by (aisle, section, shelf) -> list of SKUs in sequence order
        self.shelf_index = defaultdict(list)
        for _, row in self.planogram.iterrows():
            key = (row['aisle_number'], row['section_cd'], row['shelf_number'])
            self.shelf_index[key].append({
                'sku': row['sku_number'],
                'sequence': row['plano_sequence_number'],
                'plano': row['plano_number']
            })
        
        # Sort each shelf by sequence number
        for key in self.shelf_index:
            self.shelf_index[key].sort(key=lambda x: x['sequence'])
        
        # Index by SKU for quick lookups
        self.sku_index = self.planogram.set_index('sku_number').to_dict('index')
    
    def match_row_with_anchor(
        self, 
        detected_labels: List[Dict],  # [{'position': 0, 'ocr_text': '...', 'sku': None or 'ABC123'}, ...]
        anchor_sku: str,
        anchor_position: int
    ) -> List[Dict]:
        """
        Given a row of detected labels and one confirmed SKU (anchor),
        populate the rest using planogram sequence.
        """
        if anchor_sku not in self.sku_index:
            return detected_labels
        
        # Get the shelf context from anchor
        anchor_info = self.sku_index[anchor_sku]
        shelf_key = (
            anchor_info['aisle_number'],
            anchor_info['section_cd'],
            anchor_info['shelf_number']
        )
        
        shelf_skus = self.shelf_index.get(shelf_key, [])
        if not shelf_skus:
            return detected_labels
        
        # Find anchor's sequence position in planogram
        anchor_seq_idx = None
        for idx, item in enumerate(shelf_skus):
            if item['sku'] == anchor_sku:
                anchor_seq_idx = idx
                break
        
        if anchor_seq_idx is None:
            return detected_labels
        
        # Map detected positions to planogram sequences
        # anchor_position in detection corresponds to anchor_seq_idx in planogram
        results = []
        for label in detected_labels:
            if label.get('sku'):  # Already identified
                results.append(label)
                continue
            
            # Calculate expected planogram index
            position_offset = label['position'] - anchor_position
            expected_seq_idx = anchor_seq_idx + position_offset
            
            if 0 <= expected_seq_idx < len(shelf_skus):
                label_copy = label.copy()
                label_copy['sku'] = shelf_skus[expected_seq_idx]['sku']
                label_copy['match_confidence'] = 'interpolated'
                results.append(label_copy)
            else:
                results.append(label)
        
        return results
    
    def infer_row_from_context(
        self,
        detected_labels: List[Dict],
        known_context: Dict,  # {'aisle': X, 'section': Y} from another row
        partial_texts: List[str],  # Partial OCR from price tags, etc.
        image_shelf_position: int  # Relative position in image (0=top)
    ) -> Tuple[Optional[str], List[Dict]]:
        """
        When no SKU is detected in a row, use context from other rows
        to identify the shelf and match partial text.
        """
        aisle = known_context.get('aisle')
        section = known_context.get('section')
        
        if not aisle or not section:
            return None, detected_labels
        
        # Get all shelves in this aisle/section
        candidate_shelves = []
        for key, skus in self.shelf_index.items():
            if key[0] == aisle and key[1] == section:
                candidate_shelves.append({
                    'shelf': key[2],
                    'skus': skus
                })
        
        # Sort shelves (assuming shelf numbers correspond to physical position)
        candidate_shelves.sort(key=lambda x: x['shelf'])
        
        # Score each candidate shelf based on partial text matches
        best_shelf = None
        best_score = 0
        
        for shelf_data in candidate_shelves:
            score = self._score_shelf_match(
                shelf_data['skus'], 
                partial_texts,
                len(detected_labels)
            )
            if score > best_score:
                best_score = score
                best_shelf = shelf_data
        
        if best_shelf and best_score > 0.3:  # Confidence threshold
            # Populate labels using this shelf's planogram
            return self._populate_from_shelf(
                detected_labels, 
                best_shelf['skus']
            )
        
        return None, detected_labels
    
    def _score_shelf_match(
        self, 
        shelf_skus: List[Dict], 
        partial_texts: List[str],
        num_detections: int
    ) -> float:
        """Score how well partial texts match a shelf's SKU patterns"""
        score = 0.0
        
        # Check count match (shelf has similar number of SKUs)
        count_diff = abs(len(shelf_skus) - num_detections)
        if count_diff == 0:
            score += 0.3
        elif count_diff <= 2:
            score += 0.1
        
        # Check partial text matches against SKU numbers/names
        for text in partial_texts:
            if not text:
                continue
            for sku_data in shelf_skus:
                # Fuzzy match partial text against SKU
                sku_str = str(sku_data['sku'])
                ratio = SequenceMatcher(None, text.upper(), sku_str.upper()).ratio()
                if ratio > 0.5:
                    score += ratio * 0.2
        
        return min(score, 1.0)
    
    def _populate_from_shelf(
        self, 
        detected_labels: List[Dict],
        shelf_skus: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """Map detected labels to shelf SKUs by position"""
        results = []
        
        for i, label in enumerate(detected_labels):
            label_copy = label.copy()
            if i < len(shelf_skus):
                label_copy['sku'] = shelf_skus[i]['sku']
                label_copy['match_confidence'] = 'inferred_from_context'
            results.append(label_copy)
        
        shelf_id = f"shelf_{shelf_skus[0]['sequence']}" if shelf_skus else None
        return shelf_id, results


class OnShelfAvailabilityProcessor:
    """Main processor that orchestrates the matching pipeline"""
    
    def __init__(self, planogram_df: pd.DataFrame):
        self.matcher = PlanogramMatcher(planogram_df)
        self.row_contexts = {}  # Cache successful matches for cross-row inference
    
    def process_image_detections(
        self, 
        detections: Dict[str, List]  # {'rows': [{'labels': [...], 'shelf_position': 0}, ...]}
    ) -> Dict:
        """
        Process all detections from an image.
        
        Two-pass approach:
        1. Process rows with OCR matches to establish context
        2. Use context to infer unmatched rows
        """
        results = {'rows': []}
        unmatched_rows = []
        
        # Pass 1: Process rows with at least one OCR match
        for row_idx, row_data in enumerate(detections['rows']):
            labels = row_data['labels']
            
            # Find any labels with successful OCR
            anchors = [
                (i, l) for i, l in enumerate(labels) 
                if l.get('ocr_sku') and self._validate_sku(l['ocr_sku'])
            ]
            
            if anchors:
                # Use first valid anchor to populate row
                anchor_pos, anchor_label = anchors[0]
                populated = self.matcher.match_row_with_anchor(
                    labels,
                    anchor_label['ocr_sku'],
                    anchor_pos
                )
                
                # Cache context for cross-row inference
                sku_info = self.matcher.sku_index.get(anchor_label['ocr_sku'])
                if sku_info:
                    self.row_contexts[row_idx] = {
                        'aisle': sku_info['aisle_number'],
                        'section': sku_info['section_cd'],
                        'shelf': sku_info['shelf_number']
                    }
                
                results['rows'].append({
                    'row_idx': row_idx,
                    'labels': populated,
                    'match_method': 'anchor_interpolation'
                })
            else:
                unmatched_rows.append((row_idx, row_data))
        
        # Pass 2: Infer unmatched rows using context
        if self.row_contexts and unmatched_rows:
            # Use the most common/confident context
            best_context = self._get_best_context()
            
            for row_idx, row_data in unmatched_rows:
                partial_texts = [
                    l.get('partial_ocr', '') for l in row_data['labels']
                ]
                
                shelf_id, populated = self.matcher.infer_row_from_context(
                    row_data['labels'],
                    best_context,
                    partial_texts,
                    row_data.get('shelf_position', row_idx)
                )
                
                results['rows'].append({
                    'row_idx': row_idx,
                    'labels': populated,
                    'match_method': 'context_inference' if shelf_id else 'unmatched',
                    'inferred_shelf': shelf_id
                })
        
        # Sort by original row order
        results['rows'].sort(key=lambda x: x['row_idx'])
        return results
    
    def _validate_sku(self, sku: str) -> bool:
        """Check if SKU exists in planogram"""
        return sku in self.matcher.sku_index
    
    def _get_best_context(self) -> Dict:
        """Get the most reliable context from matched rows"""
        if not self.row_contexts:
            return {}
        # For now, just return the first context
        # Could enhance with voting/confidence weighting
        return list(self.row_contexts.values())[0]


# Load your planogram
planogram_df = pd.read_csv('planogram.csv')

# Initialize processor
processor = OnShelfAvailabilityProcessor(planogram_df)

# Structure your detections
detections = {
    'rows': [
        {
            'shelf_position': 0,  # Top shelf
            'labels': [
                {'position': 0, 'ocr_sku': None, 'partial_ocr': '$2.99'},
                {'position': 1, 'ocr_sku': 'SKU12345', 'partial_ocr': '$3.49'},  # Anchor!
                {'position': 2, 'ocr_sku': None, 'partial_ocr': '$2.99'},
            ]
        },
        {
            'shelf_position': 1,  # Second shelf - no OCR matches
            'labels': [
                {'position': 0, 'ocr_sku': None, 'partial_ocr': '$4.'},
                {'position': 1, 'ocr_sku': None, 'partial_ocr': '99'},
                {'position': 2, 'ocr_sku': None, 'partial_ocr': '$5.49'},
            ]
        }
    ]
}

# Process
results = processor.process_image_detections(detections)

