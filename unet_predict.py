import os
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
from time import ctime
import arcpy
from arcpy import env

# ------------------------------------------
# TIF Saver Function
# ------------------------------------------

def save_prediction_as_tif(prediction, reference_file, output_file):
    """
    Saves NumPy array canopy height prediction as TIF file.

    Args:
        prediction (numpy.ndarray): Predicted canopy height array.
        reference_file (str): Path to reference image for spatial metadata.
        output_file (str): Path to save the output TIF.
    """
    # Remove batch and channel dimensions if present
    if prediction.ndim == 4:
        prediction = prediction[0, :, :, 0]
    elif prediction.ndim == 3:
        prediction = prediction[:, :, 0]

    # Open reference image to get spatial metadata
    with rasterio.open(reference_file) as src:
        profile = src.profile

    # Update metadata for single-band raster
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    # Save prediction as TIF
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)


# ------------------------------------------
# Prediction Function
# ------------------------------------------

def make_predictions(model, tile_file_paths: list[str], pred_dir: str):
    """
    Makes canopy height predictions using trained U-Net model.

    Args:
        model (tf.keras.Model): Trained U-Net model.
        tile_file_paths (list[str]): List of paths to input image tiles.
        pred_dir (str): Directory to save output canopy height predictions.
    """
    
    os.makedirs(pred_dir, exist_ok=True)

    for tile_path in tile_file_paths:
        # Load and preprocess NAIP image
        with rasterio.open(tile_path) as naip:
            naip_data = naip.read()
            naip_data = np.transpose(naip_data, (1, 2, 0))  # Convert to HWC
            naip_data = naip_data / 255.0  # Normalize
            naip_data = naip_data.astype(np.float32)

        # Predict
        prediction = model.predict(naip_data[np.newaxis, ...])

        # Save output as TIF
        base_filename = os.path.basename(tile_path)
        out_file = os.path.join(pred_dir, f"pred_{base_filename}")
        save_prediction_as_tif(prediction, tile_path, out_file)


# ------------------------------------------
# Mosaic Tiles Function
# ------------------------------------------

def mosaic_tiles(pred_dir):
    """
    Mosaics predicted tiles into area-wide rasters using ArcPy.

    Args:
        pred_dir (str): Directory containing predicted TIF files.
    """
    
    # All overwrites
    env.overwriteOutput = True

    # Output directory
    mosaic_dir = os.path.join(pred_dir, "mosaics")
    os.makedirs(mosaic_dir, exist_ok=True)

    # Extract unique area identifiers from filenames and loop through them
    tifs = [x for x in os.listdir(pred_dir) if x.endswith(".tif")]
    areas = sorted(set("_".join(x.split("_")[1:3]) for x in tifs))
    for i, area in enumerate(areas):

        # Print status
        print(f"{ctime()} {area} ({i + 1}/{len(areas)})")

        # Get TIFs for this area
        area_tifs = [os.path.join(pred_dir, x) for x in tifs if area in x]

        # Mosaic them
        arcpy.management.MosaicToNewRaster(
            input_rasters=area_tifs,
            output_location=mosaic_dir,
            raster_dataset_name_with_extension=f"chm_pred_{area}.tif",
            pixel_type="32_BIT_FLOAT",
            number_of_bands=1,
            mosaic_method="BLEND"
        )

# ------------------------------------------
# Main Block
# ------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions with U-Net model and mosaic tiles.")
    parser.add_argument("--model_file", required=True, help="Path to the U-Net model (.h5 file).")
    parser.add_argument("--input_dir", help="Directory containing input image chips (e.g., .tif files). Used if not providing explicit tile paths to a tiler script.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output predictions and mosaics.")

    args = parser.parse_args()

    model = load_model(args.model_file)

    # If an input_dir is provided, list .tif files and run predictions
    # This maintains compatibility with the original script's behavior
    # when not used as part of the larger tiling workflow.
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' not found.")
        else:
            tile_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".tif")]
            if not tile_files:
                print(f"No .tif files found in {args.input_dir}")
            else:
                make_predictions(model, tile_files, args.output_dir)
                # Mosaic tiled predictions if predictions were made
                mosaic_tiles(args.output_dir)
    else:
        print("No input_dir provided. The script expects 'make_predictions' to be called by another script (e.g., a tiling script) with a list of tile paths.")
        # Example of how it might be called by a tiling script (commented out):
        # geotiff_tiler_output_paths = ["path/to/tile1.tif", "path/to/tile2.tif"] 
        # make_predictions(model, geotiff_tiler_output_paths, args.output_dir)
        # mosaic_tiles(args.output_dir) # Mosaic after predictions

    print(f"{ctime()} All done")
