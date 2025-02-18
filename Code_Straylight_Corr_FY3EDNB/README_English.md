### Software User Manual: FY-3E MERSI L1B HDF Image Processing Tool

#### 1. Environment and Library Requirements

This software requires the following environment and libraries. Please ensure that the necessary environment is set up before running the program.

**Environment Requirements:**
- Python 3.9 or higher version

**Required Libraries:**
- numpy
- pylab
- matplotlib
- h5py
- os
- pandas
- time
- re
- math
- datetime
- cv2
- shutil
- scipy
- copy
- PIL
- pywt
- cartopy

You can install the necessary libraries using the following command (recommended to use `pip`):
```bash
pip install numpy pylab matplotlib h5py pandas opencv-python scipy pillow pywt cartopy
```

#### 2. Required Input Files

- **FY-3E L1B HDF File:** The HDF file containing the satellite sensor data (e.g., `FY3E_L1B_20230511_1300.hdf`).
- **FY-3E GEO HDF File:** The HDF file containing geographic positioning data (e.g., `FY3E_GEO_20230511_1300.hdf`).

#### 3. Main Program: `hdf5_fy3e_main_module.py`

This program processes FY-3E L1B HDF data for a specific time and generates corresponding images and processing results.

**Execution Flow:**
1. Open the main program `hdf5_fy3e_main_module.py` and modify the relevant parameters as needed.
2. The key parameters are located at lines 87 to 93 in the code:
   - `date`: The specific time to process (format: `YYYYMMDD_HHMM`, e.g., `"20230511_1300"`).
   - `rad_file`: The absolute path of the FY-3E L1B HDF file corresponding to the time.
   - `geo_file`: The absolute path of the FY-3E GEO HDF file corresponding to the time.
   - `outdir`: The folder where the results will be saved (the program will create a subfolder for each time).
   - `draw_flag`: Image drawing mode:
     - `0`: Do not draw images.
     - `1`: Draw geographic map (with map).
     - `2`: Draw rectangular flowchart.
   - `dpi0`: Image resolution for the geographic map (only applies when `draw_flag=1`).

**Generated Files:**

For example, for `20230511_1300`, if `draw_flag=1` (draw geographic map), the generated files include:
- `20230511_1300_geo.png`: Original FY-3E MERSI LLB radiance image with geographic map.
- `20230511_1300_no_stray_light.png`: Image after stray light removal.
- `20230511_1300_enhanced.png`: Image after enhancement and stripe removal.
- `FY3E_MERSI_LL_Clear_Image_20230511_1300.hdf5`: Processed HDF5 file containing the following data:
  - Latitude
  - Longitude
  - Solar Zenith Angle (SZA)
  - Stray light removed radiance or grayscale matrix (No_stray_light_image)
  - Enhanced image (Enhanced_image)

**Additional Notes:**
1. `No_stray_light_image` contains an image type label (`no stray light images`, `partial stray light images`, `common stray light images`), and processing is done based on the image type.
2. The image matrix size is `2000x1536`, consistent with the original file.
3. If the nighttime (SZA > 100°) occupies less than 30% of the image or if the invalid 0-value pixels in a fully dark image exceed 20%, processing will not be performed, and the message will display: "night proportion less 0.3 !no handle!" or "too much 0 in all dark pict !no handle!".
4. For common stray light images, if the best fit cannot be found after the maximum number of iterations (500) and gradually increasing the background threshold, fitting will be skipped, and the program will continue with the next step, displaying the message: “too long time!!!!”.

#### 4. Generated Files Example

For `20230511_1300` with `draw_flag=2` (rectangular flowchart), the generated files include:
- `20230511_1300.png`: Original FY-3E MERSI LLB radiance image.
- `20230511_1300_gauss.png`: Radiance image after Gaussian fitting to remove bright spots.
- `20230511_1300_brightness.png`: Stray light removed image after brightness balancing.
- `20230511_1300_correction.png`: Image after enhancement and stripe removal.
- `FY3E_MERSI_LL_Clear_Image_20230511_1300.hdf5`: Processed HDF5 file (same as described above).

If `SZA > 100°` and the image contains 0 values, a `YYYYMMDD_HHMM_filldark.png` image will be generated. For partial stray light images, a `YYYYMMDD_HHMM_dehaze.png` image will be generated after dehazing the image.

For images without stray light, no images with `_gauss` or `_brightness` suffixes will be generated (e.g., `outcome_2/20220613_1945`).

#### 5. Notes

**When Reading the Generated HDF5:**
- The original data contains invalid values (0) in columns `[0, 1, 2, ..., 1535]`. These invalid values should be excluded when plotting the image.
- For valid image display, ensure that SZA > 100°. Other ranges should be set to null.
- The folders Output and fy3emersill_input contain the test data

#### 6. Other Auxiliary Programs

- `draw_100aoi_pict.py`: Draws the original polluted image satisfying the SZA > 100° condition (without a map).
- `no_stray_judge.py`: Judges whether an image has stray light pollution.
- `gauss_fitting.py`: Uses a Gaussian model to fit and remove bright spots from polluted images.
- `stray_light_handle.py`: Processes stray light in different types of images.
- `algorithm_details.py`: Includes several algorithms for processing stray light images, such as inherent enhancement and stripe removal, brightness unification for partial stray light, dehazing for partial stray light, and illumination correction for common stray light.
- `resize_pict.py`: Resizes images to `2000x1536`.
- `draw_geo_pict.py`: Draws geographic images if `draw_flag=1`, including the original polluted image, the Gaussian-fitted image, and the final enhanced image.

#### 7. Summary

This toolset integrates the processing and analysis of satellite images and is suitable for handling FY-3E satellite L1B data for stray light removal, enhancement, stripe removal, and other processing. It generates high-quality images and detailed HDF5 files, making it easier for subsequent analysis and research.