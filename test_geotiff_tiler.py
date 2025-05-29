import unittest
import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Import the function to be tested
from geotiff_tiler import create_tiles

class TestCreateTiles(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories and files for testing."""
        self.test_dir = "test_temp_data"
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directories and files after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_geotiff(self, filename, width, height, num_bands=1, dtype='uint8'):
        """Helper function to create a dummy GeoTIFF file."""
        filepath = os.path.join(self.input_dir, filename)
        transform = from_origin(0, height, 1, 1) # Simple 1-meter resolution
        if num_bands == 1:
            data = np.zeros((height, width), dtype=dtype)
        else:
            data = np.zeros((num_bands, height, width), dtype=dtype)
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=num_bands,
            dtype=dtype,
            crs='EPSG:32632', # A sample CRS
            transform=transform,
        ) as dst:
            if num_bands == 1:
                dst.write(data, 1)
            else:
                dst.write(data)
        return filepath

    def test_perfect_division(self):
        """Test tiling when image dimensions are perfectly divisible by tile_size."""
        img_width, img_height = 512, 512
        tile_size = 256
        input_tiff_path = self._create_dummy_geotiff("perfect.tif", img_width, img_height)

        create_tiles(input_tiff_path, self.output_dir, tile_size)

        # Verify number of tiles
        expected_num_tiles_x = img_width // tile_size
        expected_num_tiles_y = img_height // tile_size
        expected_total_tiles = expected_num_tiles_x * expected_num_tiles_y
        
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith(".tif")]
        self.assertEqual(len(output_files), expected_total_tiles, "Incorrect number of tiles created.")

        # Verify dimensions and geotransform of a sample tile (e.g., tile_0_0.tif)
        # Tile names are like tile_{col_offset}_{row_offset}.tif
        sample_tile_path = os.path.join(self.output_dir, "tile_0_0.tif")
        self.assertTrue(os.path.exists(sample_tile_path), "Sample tile tile_0_0.tif not found.")

        with rasterio.open(sample_tile_path) as src:
            self.assertEqual(src.width, tile_size, "Tile width is incorrect.")
            self.assertEqual(src.height, tile_size, "Tile height is incorrect.")
            # Original transform: | 1.00, 0.00, 0.00|
            #                     | 0.00,-1.00, 512.00|
            #                     | 0.00, 0.00, 1.00|
            # For tile_0_0, top-left should be same as original image's top-left (0, 512)
            expected_transform = from_origin(0, img_height, 1, 1) # Origin (0,0) in pixel space, (0, img_height) in geo space
            self.assertEqual(src.transform, expected_transform, "Tile geotransform is incorrect.")

        # Verify another tile (e.g. tile_256_256.tif)
        sample_tile_path_2 = os.path.join(self.output_dir, f"tile_{tile_size}_{tile_size}.tif")
        self.assertTrue(os.path.exists(sample_tile_path_2), f"Sample tile tile_{tile_size}_{tile_size}.tif not found.")
        with rasterio.open(sample_tile_path_2) as src:
            self.assertEqual(src.width, tile_size, "Tile width for tile_256_256 is incorrect.")
            self.assertEqual(src.height, tile_size, "Tile height for tile_256_256 is incorrect.")
            # For tile_256_256, top-left x is 256*1 = 256, top-left y is 512 - 256*1 = 256 (geo space)
            expected_transform_2 = from_origin(tile_size * 1, img_height - (tile_size *1) , 1, 1)
            self.assertEqual(src.transform, expected_transform_2, "Tile geotransform for tile_256_256 is incorrect.")


    def test_imperfect_division(self):
        """Test tiling when image dimensions are not perfectly divisible by tile_size."""
        img_width, img_height = 600, 400
        tile_size = 256
        input_tiff_path = self._create_dummy_geotiff("imperfect.tif", img_width, img_height)

        create_tiles(input_tiff_path, self.output_dir, tile_size)

        # Verify number of tiles
        expected_num_tiles_x = (img_width + tile_size - 1) // tile_size # Ceiling division
        expected_num_tiles_y = (img_height + tile_size - 1) // tile_size
        expected_total_tiles = expected_num_tiles_x * expected_num_tiles_y
        
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith(".tif")]
        self.assertEqual(len(output_files), expected_total_tiles, "Incorrect number of tiles for imperfect division.")

        # Verify dimensions of a full tile (tile_0_0.tif)
        full_tile_path = os.path.join(self.output_dir, "tile_0_0.tif")
        self.assertTrue(os.path.exists(full_tile_path))
        with rasterio.open(full_tile_path) as src:
            self.assertEqual(src.width, tile_size, "Full tile width is incorrect.")
            self.assertEqual(src.height, tile_size, "Full tile height is incorrect.")
            expected_transform_full = from_origin(0, img_height, 1, 1)
            self.assertEqual(src.transform, expected_transform_full, "Full tile geotransform is incorrect.")

        # Verify dimensions of a partial tile at the right edge (tile_512_0.tif)
        # (600 width, tile_size 256. Tiles at x=0, x=256, x=512)
        # Last tile starts at x=512, width should be 600-512 = 88
        partial_right_tile_path = os.path.join(self.output_dir, "tile_512_0.tif")
        self.assertTrue(os.path.exists(partial_right_tile_path), "Partial right tile not found.")
        with rasterio.open(partial_right_tile_path) as src:
            # The geotiff_tiler creates tiles with dimensions equal to tile_size,
            # even if the actual data content is smaller for partial tiles.
            # The nodata values will fill the rest of the tile.
            self.assertEqual(src.width, tile_size, "Partial right tile width should be tile_size (metadata).")
            self.assertEqual(src.height, tile_size, "Partial right tile height should be tile_size (metadata).")
            expected_transform_partial_right = from_origin(2 * tile_size, img_height, 1, 1) # X_coord of this tile, Y_coord of original image top
            self.assertEqual(src.transform, expected_transform_partial_right, "Partial right tile geotransform is incorrect.")

        # Verify dimensions of a partial tile at the bottom edge (tile_0_256.tif)
        # (400 height, tile_size 256. Tiles at y=0, y=256)
        # Last tile starts at y_offset=256 (pixel space)
        partial_bottom_tile_path = os.path.join(self.output_dir, "tile_0_256.tif")
        self.assertTrue(os.path.exists(partial_bottom_tile_path), "Partial bottom tile not found.")
        with rasterio.open(partial_bottom_tile_path) as src:
            self.assertEqual(src.width, tile_size, "Partial bottom tile width should be tile_size (metadata).")
            self.assertEqual(src.height, tile_size, "Partial bottom tile height should be tile_size (metadata).")
            # The Y coordinate for the transform is the top of that tile.
            # Original image top-left Y is img_height (geo).
            # Tile at row_offset 256 means its top is at geo Y = img_height - 256
            expected_transform_partial_bottom = from_origin(0, img_height - (1 * tile_size), 1, 1)
            self.assertEqual(src.transform, expected_transform_partial_bottom, "Partial bottom tile geotransform is incorrect.")

        # Verify dimensions and transform of a partial tile at the bottom-right corner (tile_512_256.tif)
        partial_br_tile_path = os.path.join(self.output_dir, "tile_512_256.tif")
        self.assertTrue(os.path.exists(partial_br_tile_path), "Partial bottom-right tile not found.")
        with rasterio.open(partial_br_tile_path) as src:
            self.assertEqual(src.width, tile_size, "Partial BR tile width should be tile_size (metadata).")
            self.assertEqual(src.height, tile_size, "Partial BR tile height should be tile_size (metadata).")
            expected_transform_partial_br = from_origin(2 * tile_size, img_height - (1 * tile_size), 1, 1)
            self.assertEqual(src.transform, expected_transform_partial_br, "Partial BR tile geotransform is incorrect.")

if __name__ == '__main__':
    unittest.main()
