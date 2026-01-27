import os
import argparse
import csv
import random
from collections import defaultdict

def generate_csv(source_dir, output_csv_path, max_slides):
    # Prepare Output Path and create directories if needed
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    # start Scanning source (Step 1: Collection)
    print(f"Scanning: {source_dir}")
    print(f"Target:   {output_csv_path}")
    if max_slides:
        print(f"Max Cap:  {max_slides} slides per patient")
    print("-" * 30)

    # Dictionary to group slides by patient:  { case_id: [ list_of_file_info ] }
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
    print("Applying caps and generating labels...")
    print("-" * 30)

   
    final_count = 0
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['case_id', 'slide_id', 'full_path', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # # Sort keys to ensure reproducibility if we re-run
        # sorted_case_ids = sorted(patient_registry.keys())

        for case_id in patient_registry.keys():
            slides = patient_registry[case_id]
            
            # A. Apply Max Cap
            if max_slides and len(slides) > max_slides:
                # Randomly select N slides
                selected_slides = random.sample(slides, max_slides)
            else:
                # Keep all slides
                selected_slides = slides

            # B. Generate Random Label (Per Patient)
            # We assign the same label to the whole patient (Standard MIL requirement)
            label = random.choice(["tumor_recurred", "didn't_recurr"])

            # C. Write Rows
            for slide in selected_slides:
                writer.writerow({
                    'case_id': case_id,
                    'slide_id': slide['slide_id'],
                    'full_path': slide['full_path'],
                    'label': label
                })
                final_count += 1

    print(f"Done! Final dataset has {final_count} slides.")
    print(f"Manifest saved to: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV manifest for SVS slides with caps and labels.")
    
    # Required Arguments
    parser.add_argument("source_path", help="Path to the source directory to scan")
    parser.add_argument("output_csv_path", help="Full path to the result CSV file")

    # Optional Arguments
    parser.add_argument("--max_slides", type=int, default=None, help="Maximum number of slides allowed per patient (randomly sampled if exceeded)")

    args = parser.parse_args()

    # Convert to absolute paths
    abs_source = os.path.abspath(args.source_path)
    abs_output = os.path.abspath(args.output_csv_path)

    generate_csv(abs_source, abs_output, args.max_slides)