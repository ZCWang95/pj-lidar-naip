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

def make_predictions(model, test_dir, pred_dir):
    """
    Makes canopy height predictions using trained U-Net model.

    Args:
        model (tf.keras.Model): Trained U-Net model.
        test_dir (str): Directory of input NAIP image chips.
        pred_dir (str): Directory to save output canopy height predictions.
    """
    
    # List and loop through test NAIP image files
    test_files = [x for x in sorted(os.listdir(test_dir)) if x.endswith(".tif")]
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)

        # Load and preprocess NAIP image
        with rasterio.open(test_path) as naip:
            naip_data = naip.read()
            naip_data = np.transpose(naip_data, (1, 2, 0))  # Convert to HWC
            naip_data = naip_data / 255.0  # Normalize
            naip_data = naip_data.astype(np.float32)

        # Predict
        prediction = model.predict(naip_data[np.newaxis, ...])

        # Save output as TIF
        out_file = test_path.replace(test_dir, pred_dir)
        save_prediction_as_tif(prediction, test_path, out_file)


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

    # Define parameters
    test_dir = "G:\\pj_cnn_naip\\data\\test\\naip"
    pred_dir = "G:\\pj_cnn_naip\\modeling\\predictions"
    model_file = "S:\\ursa\\campbell\\pj_cnn_naip\\modeling\\unet_model_v2.h5"

    # Make predictions on test NAIP image chips
    model = load_model(model_file)
    make_predictions(test_dir, pred_dir)

    # Mosaic tiled predictions
    mosaic_tiles(pred_dir)
