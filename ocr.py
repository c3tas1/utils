import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr

def improve_ocr_performance(image_path):
    # 1. PREPROCESSING: Add border
    # PaddleOCR often fails if text is right against the image edge.
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Add a 50px white border around the image
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 2. INITIALIZE MODEL WITH TUNED PARAMETERS
    # These are the specific knobs to turn for "human readable" but "machine missed" text.
    ocr = PaddleOCR(
        use_angle_cls=True,          # Essential if text is slightly rotated
        lang='en',                   # Change if you are processing other languages
        
        # --- DETECTION TUNING ---
        # If boxes are missing, LOWER these values.
        det_db_thresh=0.2,           # Default is 0.3. Low contrast text needs lower val.
        det_db_box_thresh=0.4,       # Default is 0.6. Filters out low-confidence boxes.
        
        # If boxes are too tight (cutting off first/last letter), INCREASE this.
        det_db_unclip_ratio=2.0,     # Default is 1.5. Expands the box slightly.
        
        # If image is high-res (4k+), INCREASE this to avoid downscaling/loss of detail.
        det_limit_side_len=1280,      # Default is 960.
        
        # --- RECOGNITION TUNING ---
        # If using server/GPU, you can swap to the 'server' models here by pointing 
        # to the specific model directories (det_model_dir, rec_model_dir) usually 
        # required for the highest accuracy server-side V4 models.
        # For now, we stick to parameter tuning on the standard model.
    )

    # 3. RUN INFERENCE
    result = ocr.ocr(img, cls=True)

    # 4. VISUALIZE RESULTS
    # This block separates the boxes and scores for visualization
    if result[0] is None:
        print("No text detected.")
        return

    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # Print raw text for debugging
    print("\n--- Extracted Text & Confidence ---")
    for t, s in zip(txts, scores):
        print(f"[{s:.2f}] {t}")

    # Draw the results
    # Note: You may need to provide a path to a .ttf font file for 'font_path' 
    # if you are visualizing non-Latin characters.
    im_show = draw_ocr(img, boxes, txts, scores, font_path=None)
    im_show = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)

    # Display using Matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(im_show)
    plt.axis('off')
    plt.title("PaddleOCR Result (Border Added + Tuned Thresholds)")
    plt.show()

# Replace with your image path
# improve_ocr_performance('path/to/your/image.jpg')
