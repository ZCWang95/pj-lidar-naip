import os
import argparse
import shutil
from tensorflow.keras.models import load_model
import glob # For finding files

# Custom module imports
from geotiff_tiler import create_tiles
from unet_predict import make_predictions, mosaic_tiles

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
    # The mosaic_tiles function saves its output in a 'mosaics' subdirectory within the predicted_tiles_dir
    print("Mosaicking started...")
    mosaic_tiles(predicted_tiles_dir) 
    print("Mosaicking complete.")
    
    final_mosaic_dir = os.path.join(predicted_tiles_dir, "mosaics")
    print(f"Final mosaics are located in: {final_mosaic_dir}")


    # --- 8. Cleanup ---
    if not args.keep_temp_files:
        print("Cleaning up temporary files...")
        
        # Delete temporary input tiles directory
        try:
            shutil.rmtree(temp_input_tiles_dir)
            print(f"Removed temporary input tiles directory: {temp_input_tiles_dir}")
        except OSError as e:
            print(f"Error removing {temp_input_tiles_dir}: {e.strerror}")

        # Delete individual predicted tile files (not the 'mosaics' subdirectory)
        individual_predicted_files = glob.glob(os.path.join(predicted_tiles_dir, "*.tif"))
        for f_path in individual_predicted_files:
            try:
                os.remove(f_path)
                print(f"Removed predicted tile: {f_path}")
            except OSError as e:
                print(f"Error removing {f_path}: {e.strerror}")
        
        print("Cleanup complete.")
    else:
        print("Temporary files will be kept as per --keep-temp-files flag.")

    print("Process complete.")

if __name__ == "__main__":
    main()
