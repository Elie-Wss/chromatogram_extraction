#!/usr/bin/env python
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import h5py 
import matplotlib.pyplot as plt

# Import your existing functions from the repo.
from io_functions import read_chromato_and_chromato_cube 
from processing_functions import read_spectrum_from_chromato_cube 
from visualization_functions import visualizer 
from visualization_functions import chromato_to_matrix
from visualization_functions import matrix_to_chromato

# Define default parameters
rt1_window = 4
rt2_window = 0.8
# QC thresholds could be set here or be read from command-line arguments
QC_MATCH_FLAG_THRESHOLD = 1
QC_MATCH_FACTOR_THRESHOLD = 650  # example value, adjust as needed
QC_SNR_THRESHOLD = 1000.0        # example value, adjust as needed

def extract_patch(chromato_cube, center_rt, rt1_window, rt2_window, time_rn, mod_time, chromato_shape):
    """
    Extract a patch from the chromatogram cube using RT windows.
    
    Parameters:
      chromato_cube: 3D numpy array (mass, rows, cols) of the chromatogram data.
      center_rt: A tuple/list (RT1, RT2) representing the corrected retention time.
                 RT1 is in minutes and RT2 is in seconds.
      rt1_window: The window half-size in minutes for RT1.
      rt2_window: The window half-size in seconds for RT2.
      time_rn: Time range tuple used by the conversion functions.
      mod_time: Modulation time used in your processing.
      chromato_shape: The shape of the chromatogram (rows, cols).
      
    Returns:
      patch: The extracted submatrix (patch) of the chromatogram cube.
    """
    # Define the lower left and upper right RT boundaries
    # Center_rt is (RT1, RT2)
    lower_rt = np.array([center_rt[0] - rt1_window, center_rt[1] - rt2_window])
    upper_rt = np.array([center_rt[0] + rt1_window, center_rt[1] + rt2_window])
    
    # Convert the RT boundaries to matrix indices using your conversion function.
    # Assuming chromato_to_matrix takes an array of RT pairs and returns matrix coordinates.
    window = chromato_to_matrix(np.array([lower_rt, upper_rt]), time_rn, mod_time, chromato_shape)
    
    # Extract scalar indices, ensuring they are within bounds
    x_min = int(max(float(window[0][0]), 0))
    y_min = int(max(float(window[0][1]), 0))
    x_max = int(min(float(window[1][0]), chromato_shape[0]))
    y_max = int(min(float(window[1][1]), chromato_shape[1]))
    
    # Extract patch from chromatogram_cube: dimensions are assumed (mass, rows, cols)
    patch = chromato_cube[:, x_min:x_max, y_min:y_max]
    return patch

def save_patch_as_png(patch, out_path, title=""):
    """
    Save a 2D visualization of the patch (by summing over the mass axis)
    as a PNG image.
    """
    patch_img = np.sum(patch, axis=0)
    plt.figure(figsize=(5,4))
    plt.imshow(patch_img, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()

def process_sample(sample_path, master_df, h5_file, out_img_dir):
    """
    Process a single CDF sample:
      - Load the sample chromatogram and cube.
      - Find the peaks for that sample in master_df.
      - For each peak meeting QC criteria, extract patch and spectrum.
      - Save the data in h5_file and generate PNG images.
      - Return a list of metadata dictionaries.
    """
    sample_name = os.path.splitext(os.path.basename(sample_path))[0]
    print(f"Processing sample: {sample_name}")
    
    # Read sample chromatogram and cube
    try:
        chromato, time_rn, chromato_cube, sigma, mass_range = read_chromato_and_chromato_cube(sample_path)
    except Exception as e:
        print(f"Error reading {sample_path}: {e}")
        return []
    
    # Filter master_df for this sample
    sample_peaks = master_df[master_df["Sample"] == sample_name]
    metadata_list = []
    
    for idx, row in sample_peaks.iterrows():
        # Apply QC: here we assume the master CSV has 'final_match_flag', 'NIST_2' (match factor), 'intensity_noise_ratio'
        if (row.get("final_match_flag", 0) < QC_MATCH_FLAG_THRESHOLD or
            row.get("NIST_2", 0) < QC_MATCH_FACTOR_THRESHOLD or
            row.get("intensity_noise_ratio", 0) < QC_SNR_THRESHOLD):
            continue
        
        # Retrieve the corrected coordinates.
        # We assume the master CSV has stored 'global_x' and 'global_y' for the corrected coordinate.
        try:
            x = int(round(float(row.get("global_x", 0))))
            y = int(round(float(row.get("global_y", 0))))
            coord = [x, y]
        except Exception as e:
            print(f"Error retrieving coordinates for peak {row['Mol']} in {sample_name}: {e}")
            continue
        
        mass = int(row["mass"])
        peak_name = row["Mol"]
        
        # Extract patch and mass spectrum
        patch = extract_patch(chromato_cube, coord, rt1_window, rt2_window)
        spectrum = read_spectrum_from_chromato_cube(coord, chromato_cube=chromato_cube)
        
        # Generate a unique ID for this extraction
        unique_id = str(uuid.uuid4())
        
        # Save patch and spectrum in HDF5
        # In the HDF5 file, create groups/datasets indexed by unique_id
        sample_group = h5_file.require_group(sample_name)
        # Store patch: dataset name "patch_{unique_id}"
        sample_group.create_dataset(f"patch_{unique_id}", data=patch, compression="gzip")
        # Store spectrum: dataset name "spectrum_{unique_id}"
        sample_group.create_dataset(f"spectrum_{unique_id}", data=spectrum, compression="gzip")
        
        # Save a PNG image for visualization
        png_filename = f"{sample_name}_{unique_id}_patch.png"
        png_path = os.path.join(out_img_dir, png_filename)
        title = f"{peak_name} (mass: {mass})"
        save_patch_as_png(patch, png_path, title=title)
        
        # Build metadata entry
        metadata = {
            "unique_id": unique_id,
            "Sample": sample_name,
            "Mol": peak_name,
            "mass": mass,
            "global_x": x,
            "global_y": y,
            "final_match_flag": row.get("final_match_flag", 0),
            "match_factor": row.get("NIST_2", None),
            "signal_noise_ratio": row.get("intensity_noise_ratio", None),
            # We store references to the HDF5 data via unique_id (the HDF5 file itself is centralized)
            "patch_id": f"patch_{unique_id}",
            "spectrum_id": f"spectrum_{unique_id}",
            # Also store the visualization image filename for debugging purposes
            "patch_image": png_filename
        }
        metadata_list.append(metadata)
    
    return metadata_list

def main():
    parser = argparse.ArgumentParser(description="Chromatogram Patch and Mass Spectrum Extraction")
    parser.add_argument("master_csv", help="Path to the master CSV file with annotations and corrected coordinates")
    parser.add_argument("cdf_dir", help="Directory containing CDF sample files")
    parser.add_argument("output_hdf5", help="Path to the output HDF5 file for storing patches and spectra")
    parser.add_argument("output_csv", help="Path to the output metadata CSV file")
    parser.add_argument("output_img_dir", help="Directory to save patch PNG images")
    args = parser.parse_args()
    
    # Load master CSV
    master_df = pd.read_csv(args.master_csv)
    
    # Create output image directory if it doesn't exist
    os.makedirs(args.output_img_dir, exist_ok=True)
    
    # Open (or create) the HDF5 file for writing extracted data
    with h5py.File(args.output_hdf5, "w") as h5_file:
        all_metadata = []
        # Iterate over CDF files in the directory
        for filename in os.listdir(args.cdf_dir):
            if filename.lower().endswith(".cdf"):
                sample_path = os.path.join(args.cdf_dir, filename)
                sample_metadata = process_sample(sample_path, master_df, h5_file, args.output_img_dir)
                all_metadata.extend(sample_metadata)
        
        # After processing all samples, save the metadata CSV
        if all_metadata:
            meta_df = pd.DataFrame(all_metadata)
            meta_df.to_csv(args.output_csv, index=False)
            print(f"Metadata CSV saved to: {args.output_csv}")
        else:
            print("No valid extractions were made based on QC criteria.")

if __name__ == "__main__":
    main()
