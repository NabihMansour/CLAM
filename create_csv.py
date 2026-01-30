import os
import argparse
import csv
import random
import pandas as pd
from collections import defaultdict

def load_labels_map(labels_csv_path):
    """
    Reads the labels CSV and returns a dictionary: { 'case_id': 'label' }
    """
    if not labels_csv_path or not os.path.exists(labels_csv_path):
        return None

    print(f"Loading labels from: {labels_csv_path}")
    try:
        # Read CSV using pandas for robustness (handles quoting/types automatically)
        df = pd.read_csv(labels_csv_path, dtype=str)
        
        # Verify required columns exist
        if 'case_id' not in df.columns or 'label' not in df.columns:
            print("Warning: Labels CSV missing 'case_id' or 'label' columns. Skipping labels.")
            return None
            
        # Create dictionary mapping for fast O(1) lookups
        # set_index('case_id')['label'] creates a Series, to_dict() converts it
        label_map = df.set_index('case_id')['label'].to_dict()
        print(f"Loaded labels for {len(label_map)} cases.")
        return label_map

    except Exception as e:
        print(f"Error loading labels CSV: {e}")
        return None

def generate_csv(source_dir, output_csv_path, max_slides, labels_csv_path=None):
    # Prepare Output Path and create directories if needed
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    # 1. Load Labels (Optional Step)
    case_labels_map = load_labels_map(labels_csv_path)
    # Boolean flag: True if we successfully loaded a map
    include_labels = (case_labels_map is not None)

    # Start Scanning source (Step 2: Collection)
    print(f"Scanning: {source_dir}")
    print(f"Target:   {output_csv_path}")
    if max_slides:
        print(f"Max Cap:  {max_slides} slides per patient")
    print("-" * 30)

    # Dictionary to group slides by patient: { case_id: [ list_of_file_info ] }
    patient_registry = defaultdict(list)
    total_slides_found = 0

    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(".svs"):
                
                # --- IDENTIFICATION LOGIC ---
                full_path = os.path.abspath(os.path.join(root, filename))
                slide_id = os.path.splitext(filename)[0]
                
                # Determine Case ID
                if os.path.abspath(root) == os.path.abspath(source_dir):
                    case_id = slide_id  # File in root
                else:
                    case_id = os.path.basename(root) # File in subfolder

                # Add to registry
                patient_registry[case_id].append({
                    'slide_id': slide_id,
                    'full_path': full_path
                })
                total_slides_found += 1

    print(f"Found {total_slides_found} total slides across {len(patient_registry)} patients.")
    print("Applying caps and generating output...")
    print("-" * 30)

    final_count = 0
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        # Define columns dynamically based on whether labels exist
        fieldnames = ['case_id', 'slide_id', 'full_path']
        if include_labels:
            fieldnames.append('label')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for case_id in patient_registry.keys():
            slides = patient_registry[case_id]
            
            # A. Determine Label (if map exists)
            current_label = None
            if include_labels:
                if case_id in case_labels_map:
                    current_label = case_labels_map[case_id]
                else:
                    continue  # Skip this case if no label found

            # B. Apply Max Cap
            if max_slides and len(slides) > max_slides:
                selected_slides = random.sample(slides, max_slides)
            else:
                selected_slides = slides

            # C. Write Rows
            slide_count = len(selected_slides)
            print(f"Writing {slide_count} slides for case: {case_id} (Label: {current_label})")
            for slide in selected_slides:
                row_data = {
                    'case_id': case_id,
                    'slide_id': slide['slide_id'],
                    'full_path': slide['full_path']
                }
                
                # Only add the label field if we are in labeling mode
                if include_labels:
                    row_data['label'] = current_label
                
                writer.writerow(row_data)
                final_count += 1

    print(f"Done! Final dataset has {final_count} slides.")
    print(f"Manifest saved to: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV manifest for SVS slides.")
    
    # Required Arguments
    parser.add_argument("source_dir", help="Path to the source directory to scan")
    parser.add_argument("output_csv_path", help="Full path to the result CSV file")

    # Optional Arguments
    parser.add_argument("--labels_csv_path", default=None, help="Path to CSV with 'case_id' and 'label'. If omitted, no labels are added.")
    parser.add_argument("--max_slides", type=int, default=None, help="Maximum number of slides allowed per patient")

    args = parser.parse_args()

    # Convert to absolute paths
    abs_source = os.path.abspath(args.source_dir)
    abs_output = os.path.abspath(args.output_csv_path)
    
    # Handle optional label path
    abs_labels = None
    if args.labels_csv_path:
        abs_labels = os.path.abspath(args.labels_csv_path)

    generate_csv(abs_source, abs_output, args.max_slides, abs_labels)