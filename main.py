#!/usr/bin/env python
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RectBivariateSpline  # For spline interpolation

# Import your existing functions.
from io_functions import read_chromato_and_chromato_cube
from processing_functions import read_spectrum_from_chromato_cube
from visualization_functions import chromato_to_matrix, matrix_to_chromato, point_is_visible, get_cmap

# -- Default parameters --
# Full window extraction parameters (in RT units)
DEFAULT_FULL_RT1_WINDOW = 0.5  # minutes (half window width for RT1)
DEFAULT_FULL_RT2_WINDOW = 0.2  # seconds (half window width for RT2)
# Clipping parameters (in RT units; these define a smaller window centered on the peak)
DEFAULT_CLIP_RT1_WINDOW = 0.5  # minutes (half width for clipping around peak in RT1)
DEFAULT_CLIP_RT2_WINDOW = 0.2  # seconds (half width for clipping around peak in RT2)
# QC thresholds (adjust as needed)
QC_MATCH_FLAG_THRESHOLD = 1
QC_MATCH_FACTOR_THRESHOLD = 750
QC_SNR_THRESHOLD = 50.0
# Mass offset (to convert mass value to index)
DEFAULT_MASS_OFFSET = 39
# Upscaling factor for patch images (using spline interpolation)
UPSCALE_FACTOR = 4

def extract_full_window_by_rt_and_mass(chromato_cube, center_rt, full_rt1_window, full_rt2_window,
                                       mass, time_rn, mod_time, chromato_shape, mass_offset=DEFAULT_MASS_OFFSET):
    """
    Extract a full window patch (2D, from a specific mass channel) from the chromatogram cube,
    where the window is defined in RT units relative to center_rt.
    
    Returns:
      full_patch: 2D numpy array (full extracted region).
      rt_bounds: tuple of ((rt1_min, rt2_min), (rt1_max, rt2_max)) used for extraction.
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
    
    return full_patch, full_rt_bounds

def clip_patch_by_rt(full_patch, center_rt, clip_rt1_window, clip_rt2_window, full_rt_bounds):
    """
    Clip the full window patch to a smaller region around center_rt.
    
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
    clip_x_min = int(round((desired_rt1_min - full_rt1_min) / (full_rt1_max - full_rt1_min) * (height - 1)))
    clip_x_max = int(round((desired_rt1_max - full_rt1_min) / (full_rt1_max - full_rt1_min) * (height - 1))) + 1
    clip_y_min = int(round((desired_rt2_min - full_rt2_min) / (full_rt2_max - full_rt2_min) * (width - 1)))
    clip_y_max = int(round((desired_rt2_max - full_rt2_min) / (full_rt2_max - full_rt2_min) * (width - 1))) + 1
    clip_x_min = max(0, clip_x_min)
    clip_y_min = max(0, clip_y_min)
    clip_x_max = min(height, clip_x_max)
    clip_y_max = min(width, clip_y_max)
    
    clipped_patch = full_patch[clip_x_min:clip_x_max, clip_y_min:clip_y_max]
    return clipped_patch

def upscale_patch(patch, upscale_factor=UPSCALE_FACTOR):
    """
    Upscale a 2D patch using bicubic spline interpolation.
    
    Returns a new patch that has (upscale_factor) times the number of rows and columns.
    """
    height, width = patch.shape
    # Original grid:
    y = np.arange(height)
    x = np.arange(width)
    # New grid:
    new_y = np.linspace(0, height - 1, height * upscale_factor)
    new_x = np.linspace(0, width - 1, width * upscale_factor)
    spline = RectBivariateSpline(y, x, patch, kx=3, ky=3)
    highres_patch = spline(new_y, new_x)
    return highres_patch

def save_patch_as_png(patch, out_path, title="", upscale_factor=UPSCALE_FACTOR):
    """
    Save a 2D patch as a PNG image. The patch is transposed so that the x-axis corresponds to RT1.
    The patch is upscaled using spline interpolation for a higher-definition output.
    """
    # Upscale the patch using spline-based interpolation.
    highres_patch = upscale_patch(patch, upscale_factor)
    img = highres_patch.T  # transpose so that x corresponds to RT1
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(img, cmap='viridis', aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel("RT1")
    plt.ylabel("RT2")
    plt.colorbar()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def process_sample_group(sample_name, sample_rows, cdf_dir, h5_file, out_img_dir,
                         full_rt1_window, full_rt2_window,
                         clip_rt1_window, clip_rt2_window):
    """
    Process all peaks for a given sample:
      1. Open the corresponding CDF.
      2. For each peak row, extract the full window (in RT) from the correct mass channel,
         then clip that window around the peak.
      3. Also extract the mass spectrum using high-precision coordinate conversion.
      4. Save the clipped patch and spectrum in HDF5 and as a PNG.
      5. Collect metadata.
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
            continue
        
        try:
            # Get corrected RT values (RT1 in minutes, RT2 in seconds) from CSV.
            rt1_corr = float(row.get("RT1_corrected", 0))
            rt2_corr = float(row.get("RT2_corrected", 0))
            center_rt = (rt1_corr, rt2_corr)
        except Exception as e:
            print(f"Error extracting RT for {row['Mol']} in {sample_name}: {e}")
            continue
        
        mass = int(row["mass"])
        peak_name = row["Mol"]
        
        # Extract full window patch using the full RT parameters.
        try:
            full_patch, full_rt_bounds = extract_full_window_by_rt_and_mass(
                chromato_cube, center_rt,
                full_rt1_window, full_rt2_window,
                mass, time_rn, 1.25, chromato_shape
            )
        except Exception as e:
            print(f"Error extracting full window for {peak_name} in {sample_name}: {e}")
            continue
        
        # Clip the full patch to a smaller region using clipping RT window parameters.
        try:
            clipped_patch = clip_patch_by_rt(full_patch, center_rt, clip_rt1_window, clip_rt2_window, full_rt_bounds)
        except Exception as e:
            print(f"Error clipping patch for {peak_name} in {sample_name}: {e}")
            continue
        
        # Extract the mass spectrum using high-precision coordinate conversion.
        try:
            # Convert the corrected RT to matrix coordinates consistently.
            corrected_coord = np.array(chromato_to_matrix(np.array([[rt1_corr, rt2_corr]]), time_rn, 1.25, chromato_shape))[0]
            spectrum = read_spectrum_from_chromato_cube(corrected_coord, chromato_cube=chromato_cube)
        except Exception as e:
            print(f"Error extracting spectrum for {peak_name} in {sample_name}: {e}")
            continue
        
        unique_id = str(uuid.uuid4())
        sample_group_h5 = h5_file.require_group(sample_name)
        sample_group_h5.create_dataset(f"patch_{unique_id}", data=clipped_patch, compression="gzip")
        sample_group_h5.create_dataset(f"spectrum_{unique_id}", data=spectrum, compression="gzip")
        
        png_filename = f"{sample_name}_{unique_id}_patch.png"
        png_path = os.path.join(out_img_dir, png_filename)
        title = f"{peak_name} | Mass: {mass} | RT1: {rt1_corr:.2f} min | RT2: {rt2_corr:.3f} s"
        save_patch_as_png(clipped_patch, png_path, title=title, upscale_factor=UPSCALE_FACTOR)
        
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
            "patch_id": f"patch_{unique_id}",
            "spectrum_id": f"spectrum_{unique_id}",
            "patch_image": png_filename
        }
        sample_metadata.append(metadata)
    
    return sample_metadata

def main():
    parser = argparse.ArgumentParser(
        description="Extract full-window chromatogram patches (in RT units) and clip around peaks."
    )
    parser.add_argument("master_csv", help="Path to the master CSV file with annotations")
    parser.add_argument("cdf_dir", help="Directory containing CDF sample files")
    parser.add_argument("output_hdf5", help="Path to the output HDF5 file for storing patches and spectra")
    parser.add_argument("output_csv", help="Path to the output metadata CSV file")
    parser.add_argument("output_img_dir", help="Directory to save patch PNG images")
    # Full window parameters (in RT units)
    parser.add_argument("--full_rt1_window", type=float, default=DEFAULT_FULL_RT1_WINDOW,
                        help="Half window (in minutes) for full extraction")
    parser.add_argument("--full_rt2_window", type=float, default=DEFAULT_FULL_RT2_WINDOW,
                        help="Half window (in seconds) for full extraction")
    # Clipping parameters (in RT units)
    parser.add_argument("--clip_rt1_window", type=float, default=DEFAULT_CLIP_RT1_WINDOW,
                        help="Half window (in minutes) for clipping around peak")
    parser.add_argument("--clip_rt2_window", type=float, default=DEFAULT_CLIP_RT2_WINDOW,
                        help="Half window (in seconds) for clipping around peak")
    args = parser.parse_args()
    
    # (Optional) For testing, you can restrict the master CSV to the first N rows.
    master_df = pd.read_csv(args.master_csv)
    master_df = master_df.head(20)
    
    grouped = master_df.groupby("Sample")
    os.makedirs(args.output_img_dir, exist_ok=True)
    
    all_metadata = []
    with h5py.File(args.output_hdf5, "w") as h5_file:
        for sample_name, group_rows in grouped:
            sample_metadata = process_sample_group(
                sample_name, group_rows, args.cdf_dir, h5_file, args.output_img_dir,
                args.full_rt1_window, args.full_rt2_window,
                args.clip_rt1_window, args.clip_rt2_window
            )
            all_metadata.extend(sample_metadata)
    
    if all_metadata:
        meta_df = pd.DataFrame(all_metadata)
        meta_df.to_csv(args.output_csv, index=False)
        print(f"Metadata CSV saved to: {args.output_csv}")
    else:
        print("No valid extractions were made based on QC criteria.")

if __name__ == "__main__":
    main()
