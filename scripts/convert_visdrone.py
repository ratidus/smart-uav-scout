import os
import cv2
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# Base path to the VisDrone dataset directory
BASE_DATASET_DIR = "../datasets/VisDrone"

# Folders to process (Train, Validation, Test)
SUBSETS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev"
]

# VisDrone object categories to YOLO class mapping (0-indexed)
# VisDrone ignores 0 (ignored regions) and 11 (others)
CLASS_MAPPING = {
    1: 0,  # Pedestrian
    2: 1,  # People
    3: 2,  # Bicycle
    4: 3,  # Car
    5: 4,  # Van
    6: 5,  # Truck
    7: 6,  # Tricycle
    8: 7,  # Awning-tricycle
    9: 8,  # Bus
    10: 9  # Motor
}

def convert_subset(subset_name):
    """
    Converts VisDrone annotation format to YOLO format for a specific dataset subset.
    Reads bounding boxes, normalizes them, and saves them as .txt files.
    """
    subset_path = os.path.join(BASE_DATASET_DIR, subset_name)
    input_images = os.path.join(subset_path, "images")
    input_labels = os.path.join(subset_path, "annotations")
    output_labels = os.path.join(subset_path, "labels")

    # Skip if the directory doesn't exist
    if not os.path.exists(subset_path):
        print(f"[WARNING] Subset directory not found: {subset_path}. Skipping.")
        return

    os.makedirs(output_labels, exist_ok=True)

    annotation_files = [f for f in os.listdir(input_labels) if f.endswith('.txt')]
    print(f"[*] Processing '{subset_name}': Found {len(annotation_files)} files.")

    for filename in tqdm(annotation_files, desc=f"Converting {subset_name}"):
        filepath = os.path.join(input_labels, filename)
        
        # Handle potentially empty annotation files
        try:
            df = pd.read_csv(filepath, header=None)
        except pd.errors.EmptyDataError:
            # Create an empty txt file for YOLO (background image without objects)
            with open(os.path.join(output_labels, filename), 'w') as f:
                pass
            continue

        image_name = filename.replace('.txt', '.jpg')
        image_path = os.path.join(input_images, image_name)
        
        # Read image to get actual dimensions for normalization
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Image not found: {image_path}. Skipping annotation.")
            continue
            
        img_height, img_width = img.shape[:2]
        yolo_lines = []
        
        # Iterate over each detected object in the VisDrone annotation
        for _, row in df.iterrows():
            object_category = int(row[5])
            
            # Filter out ignored classes
            if object_category not in CLASS_MAPPING:
                continue
                
            yolo_class = CLASS_MAPPING[object_category]
            
            # VisDrone format: bbox_left, bbox_top, bbox_width, bbox_height
            x_min = row[0]
            y_min = row[1]
            w_abs = row[2]
            h_abs = row[3]
            
            # Convert to YOLO format: center_x, center_y, width, height (normalized)
            x_center = (x_min + w_abs / 2.0) / img_width
            y_center = (y_min + h_abs / 2.0) / img_height
            w_norm = w_abs / img_width
            h_norm = h_abs / img_height
            
            # Clamp values between 0.0 and 1.0 to prevent out-of-bounds errors in YOLO
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            # Format to 6 decimal places
            line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(line)
            
        # Write the YOLO formatted labels
        with open(os.path.join(output_labels, filename), 'w') as f:
            f.write('\n'.join(yolo_lines))

def main():
    """
    Main entry point of the script. Iterates through all predefined subsets.
    """
    print("=== VisDrone to YOLO Converter ===")
    for subset in SUBSETS:
        convert_subset(subset)
    print("=== Conversion Completed Successfully ===")

if __name__ == "__main__":
    main()