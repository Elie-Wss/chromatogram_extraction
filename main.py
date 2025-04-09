#!/usr/bin/env python
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time

# Import your existing functions.
from io_functions import read_chromato_and_chromato_cube
from processing_functions import read_spectrum_from_chromato_cube
# Import visualizer and necessary visualization helpers
from visualization_functions import chromato_to_matrix, matrix_to_chromato, point_is_visible, get_cmap, visualizer # Added visualizer

# -- Default parameters --
# Window parameters (in RT units) - used for HDF5 extraction AND visualizer hints
DEFAULT_RT1_WINDOW = 0.5  # minutes (half window width for RT1)
DEFAULT_RT2_WINDOW = 0.2  # seconds (half window width for RT2)
# Removed Clipping parameters - No longer clipping for HDF5

# QC thresholds (adjust as needed)
QC_MATCH_FLAG_THRESHOLD = 1
QC_MATCH_FACTOR_THRESHOLD = 750
QC_SNR_THRESHOLD = 50.0
# Mass offset (to convert mass value to index)
DEFAULT_MASS_OFFSET = 39

# --- Function for HDF5 data extraction ---
# Renamed from extract_full_window... to reflect its sole purpose now
def extract_patch_by_rt_and_mass(chromato_cube, center_rt, rt1_window, rt2_window,
                                 mass, time_rn, mod_time, chromato_shape, mass_offset=DEFAULT_MASS_OFFSET):
    """
    Extract a patch (2D, from a specific mass channel) from the chromatogram cube,
    where the window is defined in RT units relative to center_rt by rt1/rt2_window.
    USED FOR HDF5 DATA SAVING.
    
    Returns:
      patch_data: 2D numpy array (extracted region).
      rt_bounds: tuple of ((rt1_min, rt2_min), (rt1_max, rt2_max)) used for extraction.
      mass_index: The calculated mass index.
    """
    # Define window boundaries in RT units.
    rt1_min = center_rt[0] - rt1_window
    rt1_max = center_rt[0] + rt1_window
    rt2_min = center_rt[1] - rt2_window
    rt2_max = center_rt[1] + rt2_window
    rt_bounds = ((rt1_min, rt2_min), (rt1_max, rt2_max))

    # Convert these RT boundaries to matrix indices using the standard conversion.
    position = np.array([[rt1_min, rt2_min], [rt1_max, rt2_max]])
    window_idx = chromato_to_matrix(position, time_rn, mod_time, chromato_shape)
    # Ensure indices are within valid range and cast safely
    x_min = int(max(np.floor(window_idx[0][0]), 0))
    y_min = int(max(np.floor(window_idx[0][1]), 0))
    x_max = int(min(np.ceil(window_idx[1][0]), chromato_shape[0]))
    y_max = int(min(np.ceil(window_idx[1][1]), chromato_shape[1]))
    
    mass_index = int(mass) - mass_offset
    if mass_index < 0 or mass_index >= chromato_cube.shape[0]:
        raise ValueError(f"Computed mass index {mass_index} out of bounds for mass {mass}.")

    # Ensure indices make sense before slicing
    if x_min >= x_max or y_min >= y_max:
         print(f"Warning: Calculated indices [{x_min}:{x_max}, {y_min}:{y_max}] are invalid for RT bounds {rt_bounds}. Returning empty patch.")
         return np.array([[]]), rt_bounds, mass_index # Return empty patch

    # Extract patch for the specified mass channel.
    patch_data = chromato_cube[mass_index, x_min:x_max, y_min:y_max]

    # Handle case where slicing might still result in empty if bounds were right at the edge
    if patch_data.size == 0:
        print(f"Warning: Extracted patch is empty for indices [{x_min}:{x_max}, {y_min}:{y_max}].")

    return patch_data, rt_bounds, mass_index

# --- Removed function ---
# def clip_patch_by_rt(...) - No longer used

# --- Main processing function ---
def process_sample_group(sample_name, sample_rows, cdf_dir, h5_file, out_img_dir,
                         rt1_window, rt2_window, # Combined window parameters
                         mod_time=1.25,
                         mass_offset=DEFAULT_MASS_OFFSET
                         ):
    """
    Process all peaks for a given sample:
      1. Open the corresponding CDF.
      2. For each QUALIFYING peak row:
         a. Extract a patch based on rt1/rt2_window -> SAVE patch to HDF5.
         b. Extract the mass spectrum -> SAVE spectrum to HDF5.
         c. Call `visualizer` to plot the full mass slice with window hints -> SAVE as PNG.
      3. Collect metadata.
    """
    sample_cdf_path = os.path.join(cdf_dir, sample_name + ".cdf")
    print(f"Processing sample: {sample_name} from file {sample_cdf_path}")

    try:
        chromato, time_rn, chromato_cube, sigma, mass_range = read_chromato_and_chromato_cube(sample_cdf_path)
        # Check if chromato_cube is valid
        if chromato_cube is None or chromato_cube.size == 0:
             print(f"Error: Failed to load valid chromato_cube from {sample_cdf_path}. Skipping sample.")
             return []
    except Exception as e:
        print(f"Error reading {sample_cdf_path}: {e}")
        return []

    chromato_shape = chromato.shape  # (rows, cols)
    sample_metadata = []

    for idx, row in sample_rows.iterrows():
        # QC checks: ensure the peak meets quality criteria.
        if (row.get("final_match_flag", 0) < QC_MATCH_FLAG_THRESHOLD or
            row.get("NIST_2", 0) < QC_MATCH_FACTOR_THRESHOLD or
            row.get("intensity_noise_ratio", 0) < QC_SNR_THRESHOLD):
            continue # Skip low quality peaks

        try:
            # Get corrected RT values (RT1 in minutes, RT2 in seconds) from CSV.
            rt1_corr = float(row.get("RT1_corrected", 0)) # minutes
            rt2_corr = float(row.get("RT2_corrected", 0)) # seconds
            center_rt = (rt1_corr, rt2_corr)
            mass = int(row["mass"])
            peak_name = row["Mol"]
        except Exception as e:
            print(f"Error extracting RT/Mass for row {idx} in {sample_name}: {e}")
            continue # Skip peaks with invalid data

        # --- HDF5 Patch Data Extraction (NEW Logic: Use main window parameters) ---
        try:
            # Extract patch data using the main rt_window parameters
            patch_data_for_hdf5, rt_bounds_used, mass_index = extract_patch_by_rt_and_mass(
                chromato_cube, center_rt,
                rt1_window, rt2_window, # Use main window params
                mass, time_rn, mod_time, chromato_shape, mass_offset
            )
             # Handle case where extraction results in an empty patch
            if patch_data_for_hdf5.size == 0:
                print(f"Warning: Extracted patch for HDF5 for {peak_name} in {sample_name} is empty. Skipping HDF5 save for patch.")
                # Decide if you want to skip the entire peak or just the HDF5 patch save
                # continue # Option: skip entire peak if extracted patch is empty

        except ValueError as ve: # Catch specific error from extract function
             print(f"Error extracting patch for HDF5 for {peak_name} in {sample_name}: {ve}")
             continue # Skip this peak
        except Exception as e:
            print(f"Unexpected error extracting patch for HDF5 for {peak_name} in {sample_name}: {e}")
            continue # Skip peak if HDF5 data extraction fails

        # --- Spectrum Extraction (Original Logic) ---
        try:
            # Convert the corrected RT to matrix coordinates consistently.
            # Ensure coordinates from chromato_to_matrix are valid indices
            raw_coord = chromato_to_matrix(np.array([[rt1_corr, rt2_corr]]), time_rn, mod_time, chromato_shape)[0]
            # Use floor/ceil and clip to ensure coords are within bounds for indexing chromato_cube
            spec_coord_x = np.clip(int(np.round(raw_coord[0])), 0, chromato_cube.shape[1] - 1) # Assuming shape is (mass, x, y) -> check this!
            spec_coord_y = np.clip(int(np.round(raw_coord[1])), 0, chromato_cube.shape[2] - 1)
            # Pass indices to read_spectrum... if it expects indices, otherwise pass raw_coord if it handles floats
            # Assuming read_spectrum_from_chromato_cube handles potential float coordinates or expects indices like this:
            corrected_coord_indices = (spec_coord_x, spec_coord_y) # Example if it needs indices
            
            # Check if read_spectrum_from_chromato_cube expects coordinates or indices
            # Option 1: Pass float coordinates if it handles interpolation/rounding
            # spectrum_data = read_spectrum_from_chromato_cube(raw_coord, chromato_cube=chromato_cube) 
            # Option 2: Pass rounded/clipped indices (safer if direct indexing occurs)
            spectrum_data = read_spectrum_from_chromato_cube(corrected_coord_indices, chromato_cube=chromato_cube) # Modify based on function needs

        except Exception as e:
            print(f"Error extracting spectrum for {peak_name} in {sample_name} at coords {raw_coord}: {e}")
            spectrum_data = None # Or continue / set empty array

        # --- PNG Image Generation (Logic using visualizer) ---
        unique_id = str(uuid.uuid4())
        png_filename = f"{sample_name}_{unique_id}_vis.png"
        png_path = os.path.join(out_img_dir, png_filename)
        title = f"{peak_name} | Mass: {mass} | RT1: {rt1_corr:.2f} min | RT2: {rt2_corr:.3f} s | {sample_name}"

        try:
            print(f"Generating visualization for: {title}")
            # Call the visualizer function, passing the full mass slice and window hints
            if 0 <= mass_index < chromato_cube.shape[0]:
                visualizer(
                    (chromato_cube[mass_index, :, :], time_rn), # Pass full 2D slice for the mass
                    title=title,
                    log_chromato=False,
                    rt1=rt1_corr,       # RT1 in minutes (center)
                    rt2=rt2_corr,       # RT2 in seconds (center)
                    # Pass RT window sizes (using the main window args)
                    rt1_window=rt1_window, # In minutes
                    rt2_window=rt2_window, # In seconds
                    mod_time=mod_time,
                    show=False      # Suppress interactive display
                )
                plt.savefig(png_path, bbox_inches='tight')
                plt.close(plt.gcf())
                print(f"Saved visualization to: {png_path}")
            else:
                 # This case should be caught by the check in extract_patch_by_rt_and_mass now
                 print(f"Error: Mass index {mass_index} out of bounds. Should have been caught earlier.")
                 png_filename = None

        except Exception as e:
            print(f"Error during visualization generation for {peak_name} in {sample_name}: {e}")
            plt.close('all')
            png_filename = None

        # --- Save Data to HDF5 and Collect Metadata ---
        # Save if patch data is valid and spectrum data was obtained
        if patch_data_for_hdf5.size > 0 and spectrum_data is not None:
             try:
                sample_group_h5 = h5_file.require_group(sample_name)
                patch_id = f"patch_{unique_id}"
                spectrum_id = f"spectrum_{unique_id}"

                # Save the extracted patch (using main rt_window params)
                sample_group_h5.create_dataset(patch_id, data=patch_data_for_hdf5, compression="gzip")
                # Save the spectrum
                sample_group_h5.create_dataset(spectrum_id, data=spectrum_data, compression="gzip")

                metadata = {
                    "unique_id": unique_id,
                    "Sample": sample_name,
                    "Mol": peak_name,
                    "mass": mass,
                    "RT1_corrected": rt1_corr,
                    "RT2_corrected": rt2_corr,
                    "final_match_flag": row.get("final_match_flag", 0),
                    "match_factor": row.get("NIST_2", None),
                    "signal_noise_ratio": row.get("intensity_noise_ratio", None),
                    "patch_id": patch_id,
                    "spectrum_id": spectrum_id,
                    "patch_image": png_filename # Will be None if viz failed
                }
                sample_metadata.append(metadata)
             except Exception as e:
                  print(f"Error saving HDF5 data for {peak_name} in {sample_name}: {e}")
        else:
             print(f"Skipping metadata entry for {peak_name} due to invalid patch/spectrum data for HDF5.")

    return sample_metadata

def main():
    parser = argparse.ArgumentParser(
        description="Extract chromatogram patches/spectra (HDF5) and generate visualizations (PNG) using 'visualizer'."
    )
    parser.add_argument("master_csv", help="Path to the master CSV file with annotations")
    parser.add_argument("cdf_dir", help="Directory containing CDF sample files")
    parser.add_argument("output_hdf5", help="Path to the output HDF5 file for storing patches and spectra")
    parser.add_argument("output_csv", help="Path to the output metadata CSV file")
    parser.add_argument("output_img_dir", help="Directory to save patch PNG images (generated by visualizer)")
    # Main window parameters (used for HDF5 extraction AND visualizer hints)
    parser.add_argument("--rt1_window", type=float, default=DEFAULT_RT1_WINDOW,
                        help="Half window (minutes, RT1) for HDF5 extract AND visualizer hint") # Renamed & updated help
    parser.add_argument("--rt2_window", type=float, default=DEFAULT_RT2_WINDOW,
                        help="Half window (seconds, RT2) for HDF5 extract AND visualizer hint") # Renamed & updated help
    # Removed clipping parameters arguments
    parser.add_argument("--mod_time", type=float, default=1.25, help="Modulation time in seconds")
    parser.add_argument("--mass_offset", type=int, default=DEFAULT_MASS_OFFSET, help="Offset for converting mass to index")

    args = parser.parse_args()

    try:
        master_df = pd.read_csv(args.master_csv)
        print(f"Read {len(master_df)} rows from {args.master_csv}")
        # master_df = master_df.head(20) # Keep for testing if desired
    except FileNotFoundError:
        print(f"Error: Master CSV file not found at {args.master_csv}")
        return
    except Exception as e:
        print(f"Error reading master CSV {args.master_csv}: {e}")
        return

    grouped = master_df.groupby("Sample")
    os.makedirs(args.output_img_dir, exist_ok=True)

    all_metadata = []
    start_time = time.time()
    print("Starting processing...")
    with h5py.File(args.output_hdf5, "w") as h5_file:
        for i, (sample_name, group_rows) in enumerate(grouped, 1):
            print(f"\n--- Processing Group {i}/{len(grouped)}: Sample '{sample_name}' ---")
            sample_start_time = time.time()
            sample_metadata = process_sample_group(
                sample_name, group_rows, args.cdf_dir, h5_file, args.output_img_dir,
                args.rt1_window, args.rt2_window, # Pass unified window params
                args.mod_time, args.mass_offset
            )
            all_metadata.extend(sample_metadata)
            sample_end_time = time.time()
            print(f"--- Finished Sample '{sample_name}' in {sample_end_time - sample_start_time:.2f} seconds ---")

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    if all_metadata:
        meta_df = pd.DataFrame(all_metadata)
        try:
            meta_df.to_csv(args.output_csv, index=False)
            print(f"Metadata CSV saved to: {args.output_csv}")
            print(f"Successfully processed {len(meta_df)} peaks.")
        except Exception as e:
            print(f"Error saving metadata CSV to {args.output_csv}: {e}")
    else:
        print("No valid peaks met the criteria or processed successfully across all samples.")

if __name__ == "__main__":
    main()
