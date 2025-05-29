import rasterio
import numpy as np
import os

# This is an empty Python script for GeoTIFF tiling.
# Functions for reading and tiling GeoTIFF files will be added later.

def create_tiles(input_file: str, output_dir: str, tile_size: int):
    """
    Reads a GeoTIFF file and divides it into smaller tiles.

    Args:
        input_file: Path to the input GeoTIFF file.
        output_dir: Directory to save the output tiles.
        tile_size: The size of the tiles in pixels (e.g., 256 for 256x256 tiles).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with rasterio.open(input_file) as src:
            meta = src.meta.copy()
            for j in range(0, src.height, tile_size):
                for i in range(0, src.width, tile_size):
                    window = rasterio.windows.Window(i, j, tile_size, tile_size)
                    transform = rasterio.windows.transform(window, src.transform)

                    meta['transform'] = transform
                    meta['width'] = window.width
                    meta['height'] = window.height

                    output_filename = os.path.join(output_dir, f"tile_{i}_{j}.tif")
                    
                    with rasterio.open(output_filename, 'w', **meta) as dst:
                        dst.write(src.read(window=window))
            print(f"Successfully created tiles in {output_dir}")
    except rasterio.errors.RasterioIOError as e:
        print(f"Error: Could not open input file {input_file}")
        print(e)
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create tiles from a GeoTIFF file.")
    parser.add_argument("input_file", help="Path to the input GeoTIFF file.")
    parser.add_argument("output_dir", help="Directory to save the output tiles.")
    parser.add_argument("tile_size", type=int, help="Size of the tiles in pixels (e.g., 256).")

    args = parser.parse_args()

    create_tiles(args.input_file, args.output_dir, args.tile_size)
