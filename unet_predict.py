import os
import numpy as np
import rasterio
from rasterio.merge import merge as rasterio_merge
from tensorflow.keras.models import load_model
from time import ctime

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
# Mosaic Tiles Function (with Rasterio)
# ------------------------------------------

def mosaic_tiles_with_rasterio(predicted_tile_dir: str, output_mosaic_file: str):
    '''
    Mosaics predicted tiles from a directory into a single GeoTIFF file using Rasterio.

    Args:
        predicted_tile_dir (str): Directory containing the predicted GeoTIFF tile files.
        output_mosaic_file (str): Path to save the final mosaicked GeoTIFF.
    '''
    print(f"{ctime()} Starting rasterio mosaicking...")

    tile_files = [os.path.join(predicted_tile_dir, f) for f in os.listdir(predicted_tile_dir) if f.endswith('.tif')]

    if not tile_files:
        print(f"No .tif files found in {predicted_tile_dir} to mosaic.")
        return

    sources = []
    for tif_path in tile_files:
        try:
            sources.append(rasterio.open(tif_path))
        except rasterio.errors.RasterioIOError as e:
            print(f"Warning: Could not open {tif_path}. Skipping. Error: {e}")
            continue
    
    if not sources:
        print(f"No valid .tif files could be opened in {predicted_tile_dir}. Mosaicking aborted.")
        return

    print(f"Mosaicking {len(sources)} tiles from {predicted_tile_dir} into {output_mosaic_file}")

    mosaic, out_trans = rasterio_merge(sources)
    
    # Copy metadata from one of the source files and update for the mosaic
    out_meta = sources[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": sources[0].crs, # Assume all tiles have the same CRS
        "dtype": mosaic.dtype, # Use dtype from merged array
        "count": mosaic.shape[0], # Number of bands from merged array
        "compress": "lzw"      # Apply LZW compression
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_mosaic_file), exist_ok=True)

    # Write the mosaic to disk
    with rasterio.open(output_mosaic_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all source datasets
    for src in sources:
        src.close()
    
    print(f"{ctime()} Mosaicking complete. Output saved to {output_mosaic_file}")

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
                # When run directly, predictions are saved in output_dir.
                # The mosaic will also be saved in output_dir.
                make_predictions(model, tile_files, args.output_dir)
                
                # Define path for the output mosaic when script is run directly
                output_mosaic_path = os.path.join(args.output_dir, "mosaic_from_input_dir.tif")
                mosaic_tiles_with_rasterio(args.output_dir, output_mosaic_path)
    else:
        print("No input_dir provided. The script expects 'make_predictions' to be called by another script (e.g., a tiling script) with a list of tile paths.")
        print("If called by another script, 'mosaic_tiles_with_rasterio' should also be called by that script with appropriate paths.")
        # Example of how it might be called by predict_aoi.py (which is already implemented):
        # pred_tile_dir = "path/to/predicted_tiles"
        # final_mosaic_file = "path/to/final_aoi_mosaic.tif"
        # make_predictions(model, list_of_input_tile_paths, pred_tile_dir)
        # mosaic_tiles_with_rasterio(pred_tile_dir, final_mosaic_file)

    print(f"{ctime()} All done")
