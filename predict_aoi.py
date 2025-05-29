import os
import argparse
import shutil
from tensorflow.keras.models import load_model
import glob # For finding files

# Custom module imports
from geotiff_tiler import create_tiles
from unet_predict import make_predictions, mosaic_tiles_with_rasterio # Updated import

def main():
    parser = argparse.ArgumentParser(description="Process a large GeoTIFF: tile, predict, and mosaic.")
    parser.add_argument("--input-geotiff", required=True, help="Path to the large input GeoTIFF for AOI.")
    parser.add_argument("--output-dir", required=True, help="Base directory to save final predictions and mosaics.")
    parser.add_argument("--model-path", required=True, help="Path to the .h5 model file.")
    parser.add_argument("--tile-size", required=True, type=int, help="Integer for tile dimensions (e.g., 256).")
    parser.add_argument("--temp-dir", default=None, help="Directory for storing temporary intermediate tile files. Defaults to 'output_dir/temp_tiles'.")
    parser.add_argument("--keep-temp-files", action='store_true', help="If specified, temporary tile files will not be deleted.")

    args = parser.parse_args()

    # --- 1. Determine temporary directory for input tiles ---
    if args.temp_dir:
        temp_input_tiles_dir = args.temp_dir
    else:
        temp_input_tiles_dir = os.path.join(args.output_dir, "temp_input_tiles")
    
    os.makedirs(temp_input_tiles_dir, exist_ok=True)
    print(f"Temporary input tiles will be stored in: {temp_input_tiles_dir}")

    # --- 2. Create tiles from the input GeoTIFF ---
    print("Tiling started...")
    create_tiles(args.input_geotiff, temp_input_tiles_dir, args.tile_size)
    print("Tiling complete.")

    # --- 3. Collect generated tile paths ---
    generated_tile_paths = [os.path.join(temp_input_tiles_dir, f) for f in os.listdir(temp_input_tiles_dir) if f.endswith('.tif')]
    if not generated_tile_paths:
        print(f"No tiles were generated in {temp_input_tiles_dir}. Exiting.")
        return
    print(f"Found {len(generated_tile_paths)} tiles for prediction.")

    # --- 4. Define directory for predicted tiles ---
    predicted_tiles_dir = os.path.join(args.output_dir, "predicted_tiles")
    os.makedirs(predicted_tiles_dir, exist_ok=True)
    print(f"Predicted tiles will be stored in: {predicted_tiles_dir}")

    # --- 5. Load the Keras model ---
    print("Loading model...")
    model = load_model(args.model_path)
    print("Model loaded.")

    # --- 6. Make predictions on tiles ---
    print("Prediction started...")
    make_predictions(model, generated_tile_paths, predicted_tiles_dir)
    print("Prediction complete.")

    # --- 7. Mosaic predicted tiles ---
    print("Mosaicking started...")
    # Define the output path for the final mosaic, placing it directly in the output_dir
    # For a more unique name, you could use:
    # aoi_name = os.path.splitext(os.path.basename(args.input_geotiff))[0]
    # final_mosaic_filename = f"{aoi_name}_mosaic.tif"
    final_mosaic_filename = "final_aoi_mosaic.tif" 
    final_mosaic_path = os.path.join(args.output_dir, final_mosaic_filename)
    
    mosaic_tiles_with_rasterio(predicted_tiles_dir, final_mosaic_path)
    print("Mosaicking complete.")
    print(f"Final mosaic saved to: {final_mosaic_path}")

    # --- 8. Cleanup ---
    if not args.keep_temp_files:
        print("Cleaning up temporary files...")
        
        # Delete temporary input tiles directory
        if os.path.exists(temp_input_tiles_dir):
            try:
                shutil.rmtree(temp_input_tiles_dir)
                print(f"Removed temporary input tiles directory: {temp_input_tiles_dir}")
            except OSError as e:
                print(f"Error removing {temp_input_tiles_dir}: {e.strerror}")
        
        # Delete the directory containing individual predicted tiles (now redundant)
        if os.path.exists(predicted_tiles_dir):
            try:
                shutil.rmtree(predicted_tiles_dir)
                print(f"Removed predicted tiles directory: {predicted_tiles_dir}")
            except OSError as e:
                print(f"Error removing {predicted_tiles_dir}: {e.strerror}")
        
        print("Cleanup complete.")
    else:
        print("Temporary files (input tiles and predicted tiles) will be kept as per --keep-temp-files flag.")

    print("Process complete.")

if __name__ == "__main__":
    main()
