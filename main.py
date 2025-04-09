#!/usr/bin/env python
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
# Removed: from scipy.interpolate import RectBivariateSpline - No longer upscaling manually

# Import your existing functions.
from io_functions import read_chromato_and_chromato_cube
from processing_functions import read_spectrum_from_chromato_cube
# Import visualizer and necessary visualization helpers
from visualization_functions import chromato_to_matrix, matrix_to_chromato, point_is_visible, get_cmap, visualizer # Added visualizer

# -- Default parameters --
# Full window extraction parameters (in RT units) - used for HDF5 extraction AND visualizer hints
DEFAULT_FULL_RT1_WINDOW = 0.5  # minutes (half window width for RT1)
DEFAULT_FULL_RT2_WINDOW = 0.2  # seconds (half window width for RT2)
# Clipping parameters (in RT units; these define a smaller window centered on the peak for HDF5)
DEFAULT_CLIP_RT1_WINDOW = 0.5  # minutes (half width for clipping around peak in RT1 for HDF5)
DEFAULT_CLIP_RT2_WINDOW = 0.2  # seconds (half width for clipping around peak in RT2 for HDF5)
# QC thresholds (adjust as needed)
QC_MATCH_FLAG_THRESHOLD = 1
QC_MATCH_FACTOR_THRESHOLD = 750
QC_SNR_THRESHOLD = 50.0
# Mass offset (to convert mass value to index)
DEFAULT_MASS_OFFSET = 39
# Removed: UPSCALE_FACTOR = 4 - No longer upscaling manually

# --- Functions for HDF5 data extraction (Kept for HDF5 output) ---
def extract_full_window_by_rt_and_mass(chromato_cube, center_rt, full_rt1_window, full_rt2_window,
                                       mass, time_rn, mod_time, chromato_shape, mass_offset=DEFAULT_MASS_OFFSET):
    """
    Extract a full window patch (2D, from a specific mass channel) from the chromatogram cube,
    where the window is defined in RT units relative to center_rt.
    USED FOR HDF5 DATA SAVING.
    
    Returns:
      full_patch: 2D numpy array (full extracted region).
      rt_bounds: tuple of ((rt1_min, rt2_min), (rt1_max, rt2_max)) used for extraction.
      mass_index: The calculated mass index.
    """
    # Define full window boundaries in RT units.
    rt1_min = center_rt[0] - full_rt1_window
    rt1_max = center_rt[0] + full_rt1_window
    rt2_min = center_rt[1] - full_rt2_window
    rt2_max = center_rt[1] + full_rt2_window
    full_rt_bounds = ((rt1_min, rt2_min), (rt1_max, rt2_max))

    # Convert these RT boundaries to matrix indices using the standard conversion.
    position = np.array([[rt1_min, rt2_min], [rt1_max, rt2_max]])
    window_idx = chromato_to_matrix(position, time_rn, mod_time, chromato_shape)
    x_min = int(max(float(window_idx[0][0]), 0))
    y_min = int(max(float(window_idx[0][1]), 0))
    x_max = int(min(float(window_idx[1][0]), chromato_shape[0]))
    y_max = int(min(float(window_idx[1][1]), chromato_shape[1]))
    
    mass_index = int(mass) - mass_offset
    if mass_index < 0 or mass_index >= chromato_cube.shape[0]:
        raise ValueError(f"Computed mass index {mass_index} out of bounds.")

    # Extract full window patch for the specified mass channel.
    full_patch = chromato_cube[mass_index, x_min:x_max, y_min:y_max]

    return full_patch, full_rt_bounds, mass_index # Return mass_index as well

def clip_patch_by_rt(full_patch, center_rt, clip_rt1_window, clip_rt2_window, full_rt_bounds):
    """
    Clip the full window patch to a smaller region around center_rt.
    USED FOR HDF5 DATA SAVING.
    
    Parameters:
      full_patch   : 2D numpy array from the full window extraction.
      center_rt    : (rt1, rt2) corrected center (in RT units).
      clip_rt1_window : desired half-window (in minutes) for clipping (RT1).
      clip_rt2_window : desired half-window (in seconds) for clipping (RT2).
      full_rt_bounds : ((rt1_min, rt2_min), (rt1_max, rt2_max)) corresponding to the full_patch.
      
    Returns:
      clipped_patch: the subregion (2D numpy array) corresponding to center_rt ± clip_rt?_window.
    """
    (full_rt1_min, full_rt2_min), (full_rt1_max, full_rt2_max) = full_rt_bounds
    desired_rt1_min = max(center_rt[0] - clip_rt1_window, full_rt1_min)
    desired_rt1_max = min(center_rt[0] + clip_rt1_window, full_rt1_max)
    desired_rt2_min = max(center_rt[1] - clip_rt2_window, full_rt2_min)
    desired_rt2_max = min(center_rt[1] + clip_rt2_window, full_rt2_max)

    # full_patch dimensions and linear mapping
    height, width = full_patch.shape
    # Avoid division by zero if the patch dimension is zero or one
    if height <= 1 or width <= 1 or (full_rt1_max - full_rt1_min == 0) or (full_rt2_max - full_rt2_min == 0):
        # Return empty or original patch if mapping is not possible
        if height <= 1 or width <= 1:
            return np.array([[]]) # Return empty patch
        else: # If dimension > 1 but RT range is 0, return the original patch or handle appropriately
             return full_patch # Or perhaps return empty, depending on desired behavior

    clip_x_min = int(round((desired_rt1_min - full_rt1_min) / (full_rt1_max - full_rt1_min) * (height - 1)))
    clip_x_max = int(round((desired_rt1_max - full_rt1_min) / (full_rt1_max - full_rt1_min) * (height - 1))) + 1
    clip_y_min = int(round((desired_rt2_min - full_rt2_min) / (full_rt2_max - full_rt2_min) * (width - 1)))
    clip_y_max = int(round((desired_rt2_max - full_rt2_min) / (full_rt2_max - full_rt2_min) * (width - 1))) + 1
    
    clip_x_min = max(0, clip_x_min)
    clip_y_min = max(0, clip_y_min)
    clip_x_max = min(height, clip_x_max)
    clip_y_max = min(width, clip_y_max)

    # Ensure indices are valid before slicing
    if clip_x_min >= clip_x_max or clip_y_min >= clip_y_max:
         return np.array([[]]) # Return empty patch if indices are invalid

    clipped_patch = full_patch[clip_x_min:clip_x_max, clip_y_min:clip_y_max]
    return clipped_patch

# --- Removed functions ---
# def upscale_patch(...) - No longer used
# def save_patch_as_png(...) - Replaced by direct call to visualizer + plt.savefig

# --- Main processing function ---
def process_sample_group(sample_name, sample_rows, cdf_dir, h5_file, out_img_dir,
                         full_rt1_window, full_rt2_window, # Used for HDF5 extract AND visualizer hint
                         clip_rt1_window, clip_rt2_window, # Used ONLY for HDF5 extract
                         mod_time=1.25, # Define mod_time here or pass as arg
                         mass_offset=DEFAULT_MASS_OFFSET # Pass mass_offset
                         ):
    """
    Process all peaks for a given sample:
      1. Open the corresponding CDF.
      2. For each QUALIFYING peak row:
         a. Extract the full window (in RT) from the correct mass channel,
            then clip that window around the peak -> SAVE clipped patch to HDF5.
         b. Extract the mass spectrum using high-precision coordinate conversion -> SAVE spectrum to HDF5.
         c. Call `visualizer` to plot the full mass slice with window hints -> SAVE as PNG.
      3. Collect metadata.
    """
    sample_cdf_path = os.path.join(cdf_dir, sample_name + ".cdf")
    print(f"Processing sample: {sample_name} from file {sample_cdf_path}")

    try:
        chromato, time_rn, chromato_cube, sigma, mass_range = read_chromato_and_chromato_cube(sample_cdf_path)
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

        # --- HDF5 Data Extraction (Original Logic) ---
        try:
            # Extract full window patch for HDF5 saving & get mass index
            full_patch_data, full_rt_bounds, mass_index = extract_full_window_by_rt_and_mass(
                chromato_cube, center_rt,
                full_rt1_window, full_rt2_window, # Use FULL window params for initial extract
                mass, time_rn, mod_time, chromato_shape, mass_offset
            )
            # Clip the full patch to a smaller region for HDF5 saving
            clipped_patch_data = clip_patch_by_rt(
                 full_patch_data, center_rt, clip_rt1_window, clip_rt2_window, full_rt_bounds # Use CLIP window params
            )
             # Handle case where clipping results in an empty patch
            if clipped_patch_data.size == 0:
                print(f"Warning: Clipped patch for {peak_name} in {sample_name} is empty. Skipping HDF5 save for patch.")
                # Decide if you want to skip the entire peak or just the HDF5 patch save
                # continue # Option: skip entire peak if clipped patch is empty

        except Exception as e:
            print(f"Error extracting/clipping patch for HDF5 for {peak_name} in {sample_name}: {e}")
            continue # Skip peak if HDF5 data extraction fails

        # --- Spectrum Extraction (Original Logic) ---
        try:
            # Convert the corrected RT to matrix coordinates consistently.
            corrected_coord = np.array(chromato_to_matrix(np.array([[rt1_corr, rt2_corr]]), time_rn, mod_time, chromato_shape))[0]
            spectrum_data = read_spectrum_from_chromato_cube(corrected_coord, chromato_cube=chromato_cube)
        except Exception as e:
            print(f"Error extracting spectrum for {peak_name} in {sample_name}: {e}")
            # Decide if you want to continue without spectrum or skip peak
            spectrum_data = None # Or continue / set empty array
            # continue

        # --- PNG Image Generation (NEW Logic using visualizer) ---
        unique_id = str(uuid.uuid4())
        png_filename = f"{sample_name}_{unique_id}_vis.png" # Changed suffix slightly
        png_path = os.path.join(out_img_dir, png_filename)
        title = f"{peak_name} | Mass: {mass} | RT1: {rt1_corr:.2f} min | RT2: {rt2_corr:.3f} s | {sample_name}"

        try:
            print(f"Generating visualization for: {title}")
            # Call the visualizer function, passing the full mass slice and window hints
            # Note: Assumes visualizer handles potential errors if mass_index is out of bounds
            if 0 <= mass_index < chromato_cube.shape[0]:
                visualizer(
                    (chromato_cube[mass_index, :, :], time_rn), # Pass full 2D slice for the mass
                    title=title,
                    log_chromato=False, # As used in Script B
                    rt1=rt1_corr,       # RT1 in minutes (center)
                    rt2=rt2_corr,       # RT2 in seconds (center)
                    # Pass RT window sizes (use FULL window args, similar to Script B's fixed values)
                    rt1_window=full_rt1_window, # In minutes
                    rt2_window=full_rt2_window, # In seconds
                    mod_time=mod_time,
                    show=False      # Suppress interactive display
                )
                # Save the figure generated by visualizer
                plt.savefig(png_path, bbox_inches='tight')
                plt.close(plt.gcf()) # Close figure to free memory
                print(f"Saved visualization to: {png_path}")
            else:
                 print(f"Error: Mass index {mass_index} out of bounds for visualization. Skipping PNG generation.")
                 png_filename = None # Indicate no image was saved


        except Exception as e:
            print(f"Error during visualization generation for {peak_name} in {sample_name}: {e}")
            plt.close('all') # Close any potentially open figures on error
            png_filename = None # Indicate no image was saved


        # --- Save Data to HDF5 and Collect Metadata ---
        # Only save if HDF5 data is valid
        if clipped_patch_data.size > 0 and spectrum_data is not None:
             try:
                sample_group_h5 = h5_file.require_group(sample_name)
                patch_id = f"patch_{unique_id}"
                spectrum_id = f"spectrum_{unique_id}"
                
                # Save the *clipped* patch (original logic)
                sample_group_h5.create_dataset(patch_id, data=clipped_patch_data, compression="gzip")
                # Save the spectrum (original logic)
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
                    # png_filename will be None if visualization failed
                    "patch_image": png_filename
                }
                sample_metadata.append(metadata)
             except Exception as e:
                  print(f"Error saving HDF5 data for {peak_name} in {sample_name}: {e}")
                  # Decide how to handle HDF5 save error (e.g., skip metadata)
                  # continue
        else:
             print(f"Skipping metadata entry for {peak_name} due to invalid patch/spectrum data for HDF5.")


    return sample_metadata

def main():
    parser = argparse.ArgumentParser(
        description="Extract chromatogram patches/spectra (HDF5) and generate visualizations (PNG) using 'visualizer'." # Modified description
    )
    parser.add_argument("master_csv", help="Path to the master CSV file with annotations")
    parser.add_argument("cdf_dir", help="Directory containing CDF sample files")
    parser.add_argument("output_hdf5", help="Path to the output HDF5 file for storing patches and spectra")
    parser.add_argument("output_csv", help="Path to the output metadata CSV file")
    parser.add_argument("output_img_dir", help="Directory to save patch PNG images (generated by visualizer)") # Modified description
    # Full window parameters (used for HDF5 AND visualizer)
    parser.add_argument("--full_rt1_window", type=float, default=DEFAULT_FULL_RT1_WINDOW,
                        help="Half window (minutes, RT1) for HDF5 extract AND visualizer hint") # Modified help
    parser.add_argument("--full_rt2_window", type=float, default=DEFAULT_FULL_RT2_WINDOW,
                        help="Half window (seconds, RT2) for HDF5 extract AND visualizer hint") # Modified help
    # Clipping parameters (USED ONLY FOR HDF5)
    parser.add_argument("--clip_rt1_window", type=float, default=DEFAULT_CLIP_RT1_WINDOW,
                        help="Half window (minutes, RT1) for HDF5 clipping around peak") # Modified help
    parser.add_argument("--clip_rt2_window", type=float, default=DEFAULT_CLIP_RT2_WINDOW,
                        help="Half window (seconds, RT2) for HDF5 clipping around peak") # Modified help
    # Add mod_time and mass_offset as arguments if needed, otherwise use defaults
    parser.add_argument("--mod_time", type=float, default=1.25, help="Modulation time in seconds")
    parser.add_argument("--mass_offset", type=int, default=DEFAULT_MASS_OFFSET, help="Offset for converting mass to index")

    args = parser.parse_args()

    # Consider removing .head(20) for production runs
    try:
        master_df = pd.read_csv(args.master_csv)
        print(f"Read {len(master_df)} rows from {args.master_csv}")
        # master_df = master_df.head(20) # Keep for testing if desired
        # print("Processing only first 20 rows for testing.")
    except FileNotFoundError:
        print(f"Error: Master CSV file not found at {args.master_csv}")
        return # Exit if input file is missing
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
                args.full_rt1_window, args.full_rt2_window,
                args.clip_rt1_window, args.clip_rt2_window,
                args.mod_time, args.mass_offset # Pass mod_time and mass_offset
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
