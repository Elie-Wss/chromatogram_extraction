#!/usr/bin/env python
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
# Removed RectBivariateSpline import

# Import your existing functions.
from io_functions import read_chromato_and_chromato_cube
from processing_functions import read_spectrum_from_chromato_cube
# Re-import visualizer, remove others if not needed elsewhere
from visualization_functions import chromato_to_matrix, matrix_to_chromato, get_cmap, visualizer # Added visualizer back

# -- Default parameters --
# Full window extraction parameters (in RT units) - used for initial extraction ONLY
DEFAULT_FULL_RT1_WINDOW = 0.5  # minutes (half window width for RT1)
DEFAULT_FULL_RT2_WINDOW = 0.2  # seconds (half window width for RT2)
# Clipping parameters (in RT units) - defines the final patch size for HDF5 & visualizer view hints
DEFAULT_CLIP_RT1_WINDOW = 0.1  # minutes (half width for clipping around peak in RT1)
DEFAULT_CLIP_RT2_WINDOW = 0.1175 # seconds (half width for clipping around peak in RT2)
# Removed DEFAULT_UPSCALE_FACTOR

# QC thresholds (adjust as needed)
QC_MATCH_FLAG_THRESHOLD = 1
QC_MATCH_FACTOR_THRESHOLD = 750
QC_SNR_THRESHOLD = 50.0
# Mass offset (to convert mass value to index)
DEFAULT_MASS_OFFSET = 39

# --- Functions for data extraction and clipping ---

def extract_full_patch_by_rt_and_mass(chromato_cube, center_rt, full_rt1_window, full_rt2_window,
                                      mass, time_rn, mod_time, chromato_shape, mass_offset=DEFAULT_MASS_OFFSET):
    """
    Extracts an initial, larger patch ('full_patch') based on full_rt*_window parameters.
    """
    # (Implementation remains the same)
    rt1_min = center_rt[0] - full_rt1_window
    rt1_max = center_rt[0] + full_rt1_window
    rt2_min = center_rt[1] - full_rt2_window
    rt2_max = center_rt[1] + full_rt2_window
    full_rt_bounds = ((rt1_min, rt2_min), (rt1_max, rt2_max))

    position = np.array([[rt1_min, rt2_min], [rt1_max, rt2_max]])
    window_idx = chromato_to_matrix(position, time_rn, mod_time, chromato_shape)
    x_min = int(max(np.floor(window_idx[0][0]), 0))
    y_min = int(max(np.floor(window_idx[0][1]), 0))
    x_max = int(min(np.ceil(window_idx[1][0]), chromato_shape[0]))
    y_max = int(min(np.ceil(window_idx[1][1]), chromato_shape[1]))

    mass_index = int(mass) - mass_offset
    if mass_index < 0 or mass_index >= chromato_cube.shape[0]:
        raise ValueError(f"Computed mass index {mass_index} out of bounds for mass {mass}.")

    if x_min >= x_max or y_min >= y_max:
         print(f"Warning: Calculated indices [{x_min}:{x_max}, {y_min}:{y_max}] are invalid for initial extraction. Returning empty patch.")
         return np.array([[]]), full_rt_bounds, mass_index

    full_patch = chromato_cube[mass_index, x_min:x_max, y_min:y_max]
    if full_patch.size == 0:
        print(f"Warning: Initial extracted patch is empty for indices [{x_min}:{x_max}, {y_min}:{y_max}].")
    return full_patch, full_rt_bounds, mass_index


def clip_patch_by_rt(full_patch, full_rt_bounds, center_rt, clip_rt1_window, clip_rt2_window):
    """
    Clips the full_patch to a smaller region defined by clip_rt*_window parameters,
    mapping RT coordinates to array indices without changing resolution.
    This clipped patch is saved to HDF5.
    """
    # (Implementation remains the same)
    if full_patch.size == 0: return np.array([[]])
    (full_rt1_min, full_rt2_min), (full_rt1_max, full_rt2_max) = full_rt_bounds
    patch_height, patch_width = full_patch.shape
    desired_rt1_min = center_rt[0] - clip_rt1_window
    desired_rt1_max = center_rt[0] + clip_rt1_window
    desired_rt2_min = center_rt[1] - clip_rt2_window
    desired_rt2_max = center_rt[1] + clip_rt2_window
    target_rt1_min = max(desired_rt1_min, full_rt1_min)
    target_rt1_max = min(desired_rt1_max, full_rt1_max)
    target_rt2_min = max(desired_rt2_min, full_rt2_min)
    target_rt2_max = min(desired_rt2_max, full_rt2_max)

    if target_rt1_min >= target_rt1_max or target_rt2_min >= target_rt2_max:
        # print("Warning: Target RT range for clipping is invalid or outside the full patch bounds. Returning empty clipped patch.")
        return np.array([[]])

    rt1_range_full = full_rt1_max - full_rt1_min
    rt2_range_full = full_rt2_max - full_rt2_min
    if rt1_range_full <= 0 or rt2_range_full <= 0:
         # print("Warning: RT range of the full patch is zero. Cannot perform RT-based clipping. Returning empty.")
         return np.array([[]])

    x_clip_min_float = ((target_rt1_min - full_rt1_min) / rt1_range_full) * (patch_height -1)
    x_clip_max_float = ((target_rt1_max - full_rt1_min) / rt1_range_full) * (patch_height -1)
    y_clip_min_float = ((target_rt2_min - full_rt2_min) / rt2_range_full) * (patch_width -1)
    y_clip_max_float = ((target_rt2_max - full_rt2_min) / rt2_range_full) * (patch_width -1)

    x_clip_min_idx = int(round(x_clip_min_float))
    x_clip_max_idx = int(round(x_clip_max_float)) + 1
    y_clip_min_idx = int(round(y_clip_min_float))
    y_clip_max_idx = int(round(y_clip_max_float)) + 1

    x_clip_min_idx = max(0, x_clip_min_idx)
    y_clip_min_idx = max(0, y_clip_min_idx)
    x_clip_max_idx = min(patch_height, x_clip_max_idx)
    y_clip_max_idx = min(patch_width, y_clip_max_idx)

    if x_clip_min_idx >= x_clip_max_idx or y_clip_min_idx >= y_clip_max_idx:
        # print(f"Warning: Calculated indices [{x_clip_min_idx}:{x_clip_max_idx}, {y_clip_min_idx}:{y_clip_max_idx}] are invalid after mapping. Returning empty clipped patch.")
        return np.array([[]])

    clipped_patch = full_patch[x_clip_min_idx:x_clip_max_idx, y_clip_min_idx:y_clip_max_idx]
    # if clipped_patch.size == 0:
    #      print(f"Warning: Clipped patch is empty for slice [{x_clip_min_idx}:{x_clip_max_idx}, {y_clip_min_idx}:{y_clip_max_idx}].")
    return clipped_patch

# --- Removed functions ---
# upscale_patch
# save_clipped_patch_png

# --- Main processing function ---
def process_sample_group(sample_name, sample_rows, cdf_dir, h5_file, out_img_dir,
                         full_rt1_window, full_rt2_window, # For initial extraction ONLY
                         clip_rt1_window, clip_rt2_window, # For HDF5 clipping & visualizer hint
                         mod_time=1.25,
                         mass_offset=DEFAULT_MASS_OFFSET
                         # upscale_factor removed
                         ):
    """
    Process all peaks for a given sample:
      1. Open the corresponding CDF.
      2. For each QUALIFYING peak row:
         a. Extract an initial 'full' patch.
         b. Clip this patch based on clip_rt*_window using RT mapping -> SAVE clipped patch to HDF5.
         c. Extract the mass spectrum -> SAVE spectrum to HDF5.
         d. Call `visualizer` using CLIP window hints -> SAVE PNG visualization.
      3. Collect metadata.
    """
    sample_cdf_path = os.path.join(cdf_dir, sample_name + ".cdf")
    print(f"Processing sample: {sample_name} from file {sample_cdf_path}")

    try:
        chromato, time_rn, chromato_cube, sigma, mass_range = read_chromato_and_chromato_cube(sample_cdf_path)
        if chromato_cube is None or chromato_cube.size == 0:
             print(f"Error: Failed to load valid chromato_cube from {sample_cdf_path}. Skipping sample.")
             return []
    except Exception as e:
        print(f"Error reading {sample_cdf_path}: {e}")
        return []

    chromato_shape = chromato.shape
    sample_metadata = []

    for idx, row in sample_rows.iterrows():
        if (row.get("final_match_flag", 0) < QC_MATCH_FLAG_THRESHOLD or
            row.get("NIST_2", 0) < QC_MATCH_FACTOR_THRESHOLD or
            row.get("intensity_noise_ratio", 0) < QC_SNR_THRESHOLD):
            continue

        try:
            rt1_corr = float(row.get("RT1_corrected", 0))
            rt2_corr = float(row.get("RT2_corrected", 0))
            center_rt = (rt1_corr, rt2_corr)
            mass = int(row["mass"])
            peak_name = row["Mol"]
        except Exception as e:
            print(f"Error extracting RT/Mass for row {idx} in {sample_name}: {e}")
            continue

        # --- Patch Extraction and Clipping ---
        try:
            full_patch, full_rt_bounds, mass_index = extract_full_patch_by_rt_and_mass(
                chromato_cube, center_rt,
                full_rt1_window, full_rt2_window, # Use full windows for initial grab
                mass, time_rn, mod_time, chromato_shape, mass_offset
            )
            # Clip the patch for HDF5 saving
            clipped_patch_for_hdf5 = clip_patch_by_rt(
                 full_patch, full_rt_bounds, center_rt,
                 clip_rt1_window, clip_rt2_window # Use clip windows for clipping
            )
            if clipped_patch_for_hdf5.size == 0:
                print(f"Warning: Final clipped patch for HDF5 for {peak_name} in {sample_name} is empty.")
                # Handling below in HDF5/metadata saving block

        except ValueError as ve:
             print(f"ValueError during patch extraction/clipping for {peak_name} in {sample_name}: {ve}")
             continue
        except Exception as e:
            print(f"Unexpected error during patch extraction/clipping for {peak_name} in {sample_name}: {e}")
            continue

        # --- Spectrum Extraction ---
        try:
            raw_coord = chromato_to_matrix(np.array([[rt1_corr, rt2_corr]]), time_rn, mod_time, chromato_shape)[0]
            spec_coord_x = np.clip(int(np.round(raw_coord[0])), 0, chromato_cube.shape[1] - 1)
            spec_coord_y = np.clip(int(np.round(raw_coord[1])), 0, chromato_cube.shape[2] - 1)
            corrected_coord_indices = (spec_coord_x, spec_coord_y)
            spectrum_data = read_spectrum_from_chromato_cube(corrected_coord_indices, chromato_cube=chromato_cube)
        except Exception as e:
            print(f"Error extracting spectrum for {peak_name} in {sample_name}: {e}")
            spectrum_data = None

        # --- PNG Image Generation (using visualizer with CLIP window hints) ---
        unique_id = str(uuid.uuid4())
        # Keep filename consistent, title indicates the view parameters
        png_filename = f"{sample_name}_{unique_id}_vis.png"
        png_path = os.path.join(out_img_dir, png_filename)
        # Title reflects the view parameters (clip windows)
        title = f"View(Clip): {peak_name} | Mass: {mass} | RT1: {rt1_corr:.2f}±{clip_rt1_window:.2f} | RT2: {rt2_corr:.3f}±{clip_rt2_window:.3f} | {sample_name}"

        png_saved_successfully = False # Flag to track save status
        try:
            print(f"Generating visualization for: {title}")
            # Check mass_index is valid before accessing chromato_cube slice
            if 0 <= mass_index < chromato_cube.shape[0]:
                visualizer(
                    # Pass the full 2D slice for the specific mass
                    (chromato_cube[mass_index, :, :], time_rn),
                    title=title,
                    log_chromato=False,
                    rt1=rt1_corr,       # Peak center RT1
                    rt2=rt2_corr,       # Peak center RT2
                    # Use CLIP windows as hints for visualizer's view
                    rt1_window=clip_rt1_window,
                    rt2_window=clip_rt2_window,
                    mod_time=mod_time,
                    show=False      # Suppress interactive display
                )
                # Save the figure generated by visualizer
                plt.savefig(png_path, bbox_inches='tight', dpi=150) # Control DPI if needed
                plt.close(plt.gcf()) # Close figure
                print(f"Saved visualization to: {png_path}")
                png_saved_successfully = True
            else:
                 # This should have been caught during extraction if invalid
                 print(f"Error: Invalid mass index {mass_index} encountered before visualization.")

        except Exception as e:
            print(f"Error during visualization generation for {peak_name} in {sample_name}: {e}")
            plt.close('all') # Ensure figures are closed on error

        if not png_saved_successfully:
            png_filename = None # Set to None if saving failed

        # --- Save Data to HDF5 and Collect Metadata ---
        # Save HDF5 if clipped patch is valid (even if spectrum or PNG failed)
        if clipped_patch_for_hdf5.size > 0:
             try:
                sample_group_h5 = h5_file.require_group(sample_name)
                patch_id = f"patch_{unique_id}"
                spectrum_id = f"spectrum_{unique_id}"

                # Save the clipped patch to HDF5
                sample_group_h5.create_dataset(patch_id, data=clipped_patch_for_hdf5, compression="gzip")

                # Save spectrum only if it was successfully extracted
                if spectrum_data is not None:
                    sample_group_h5.create_dataset(spectrum_id, data=spectrum_data, compression="gzip")
                else:
                    spectrum_id = None # Indicate spectrum was not saved

                # Collect metadata, including the actual png_filename status
                metadata = {
                    "unique_id": unique_id,
                    "Sample": sample_name, "Mol": peak_name, "mass": mass,
                    "RT1_corrected": rt1_corr, "RT2_corrected": rt2_corr,
                    "final_match_flag": row.get("final_match_flag", 0),
                    "match_factor": row.get("NIST_2", None),
                    "signal_noise_ratio": row.get("intensity_noise_ratio", None),
                    "patch_id": patch_id,
                    "spectrum_id": spectrum_id, # Will be None if spectrum failed
                    "patch_image": png_filename # Will be None if PNG save failed
                }
                sample_metadata.append(metadata)

             except Exception as e:
                  print(f"Error saving HDF5 data for {peak_name} in {sample_name}: {e}")
        else:
             # Clipped patch was empty, decide if you still want to save spectrum / log metadata
             print(f"Skipping HDF5 patch save and metadata entry for {peak_name} because clipped patch was empty.")


    return sample_metadata

def main():
    parser = argparse.ArgumentParser(
        description="Extract RT-clipped patches (HDF5) / spectra and generate PNG visualizations using 'visualizer' based on clip windows." # Updated description
    )
    parser.add_argument("master_csv", help="Path to the master CSV file with annotations")
    parser.add_argument("cdf_dir", help="Directory containing CDF sample files")
    parser.add_argument("output_hdf5", help="Path to the output HDF5 file (stores RT-clipped patches)")
    parser.add_argument("output_csv", help="Path to the output metadata CSV file")
    parser.add_argument("output_img_dir", help="Directory to save PNG visualizations (view based on clip windows)") # Updated description
    # Full window parameters (initial extraction only)
    parser.add_argument("--full_rt1_window", type=float, default=DEFAULT_FULL_RT1_WINDOW,
                        help="Half window (minutes, RT1) for initial large extraction")
    parser.add_argument("--full_rt2_window", type=float, default=DEFAULT_FULL_RT2_WINDOW,
                        help="Half window (seconds, RT2) for initial large extraction")
    # Clipping parameters (final HDF5 patch size & visualizer view hints)
    parser.add_argument("--clip_rt1_window", type=float, default=DEFAULT_CLIP_RT1_WINDOW,
                        help="Half window (minutes, RT1) for HDF5 clipping and visualizer view")
    parser.add_argument("--clip_rt2_window", type=float, default=DEFAULT_CLIP_RT2_WINDOW,
                        help="Half window (seconds, RT2) for HDF5 clipping and visualizer view")
    # Removed upscale argument
    parser.add_argument("--mod_time", type=float, default=1.25, help="Modulation time in seconds")
    parser.add_argument("--mass_offset", type=int, default=DEFAULT_MASS_OFFSET, help="Offset for converting mass to index")

    args = parser.parse_args()

    try:
        master_df_full = pd.read_csv(args.master_csv) # Read the full CSV
        print(f"Read {len(master_df_full)} total rows from {args.master_csv}")

        # --- Subsetting for QC ---
        # Process only the first 18 rows for quality control
        num_rows_to_process = 18
        master_df = master_df_full.head(num_rows_to_process)
        print(f"---!!! QC RUN ACTIVE !!!--- Processing only the first {num_rows_to_process} rows from the master CSV.",flush=True)
        # --------------------------

    except FileNotFoundError:
        print(f"Error: Master CSV file not found at {args.master_csv}")
        return # Exit if input file is missing
    except Exception as e:
        print(f"Error reading master CSV {args.master_csv}: {e}")
        return

    # Add a check to see if the subset is empty
    if master_df.empty:
        print("Warning: The subset of the first {num_rows_to_process} rows is empty. No processing will occur.")
        return

    # Group the SUBSET DataFrame
    grouped = master_df.groupby("Sample")
    print(f"Found {len(grouped)} sample(s) in the first {num_rows_to_process} rows.", flush=True)

    # Create output directory
    os.makedirs(args.output_img_dir, exist_ok=True)

    

    all_metadata = []
    start_time = time.time()
    print("Starting processing...")
    with h5py.File(args.output_hdf5, "w") as h5_file:
        for i, (sample_name, group_rows) in enumerate(grouped, 1):
            print(f"\n--- Processing Group {i}/{len(grouped)}: Sample '{sample_name}' ---",flush=True)
            sample_start_time = time.time()
            sample_metadata = process_sample_group(
                sample_name, group_rows, args.cdf_dir, h5_file, args.output_img_dir,
                args.full_rt1_window, args.full_rt2_window,
                args.clip_rt1_window, args.clip_rt2_window,
                args.mod_time, args.mass_offset
                # upscale factor removed
            )
            all_metadata.extend(sample_metadata)
            sample_end_time = time.time()
            print(f"--- Finished Sample '{sample_name}' in {sample_end_time - sample_start_time:.2f} seconds ---",flush=True)

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    if all_metadata:
        meta_df = pd.DataFrame(all_metadata)
        try:
            meta_df.to_csv(args.output_csv, index=False)
            print(f"Metadata CSV saved to: {args.output_csv}")
            print(f"Successfully processed {len(meta_df)} peaks.",flush=True)
        except Exception as e:
            print(f"Error saving metadata CSV to {args.output_csv}: {e}")
    else:
        print("No valid peaks met the criteria or processed successfully across all samples.")

if __name__ == "__main__":
    main()
