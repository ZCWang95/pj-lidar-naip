import os
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import random
import itertools
import time

# ------------------------------------------
# Set Random Seed
# ------------------------------------------
RANDOM_SEED = 57

# ------------------------------------------
# Epoch Timer
# ------------------------------------------

class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        logs["epoch_time_sec"] = duration

# ------------------------------------------
# Data Generator
# ------------------------------------------

class DataGenerator(Sequence):
    """
    Custom data generator for loading NAIP and CHM images in mini-batches,
    with optional data augmentation that effectively doubles the training size.
    """

    def __init__(self, naip_dir, chm_dir, batch_size, augment=True,
                 use_sample=False, sample_proportion=0.1):
        """
        Args:
            naip_dir (str): Directory containing four-band NAIP TIF files.
            chm_dir (str): Directory containing canopy height model TIF files.
            batch_size (int): Number of samples per batch.
            augment (bool): Whether to apply data augmentation.
            use_sample (bool): Whether to use a sample of the data (e.g., for hyperparameter tuning)
            sample_proportion (float): The fractional sample of training and validation data to use
        """
        self.naip_files = sorted([
            os.path.join(naip_dir, f)
            for f in os.listdir(naip_dir)
            if f.endswith(".tif")
        ])
        self.chm_files = sorted([
            os.path.join(chm_dir, f)
            for f in os.listdir(chm_dir)
            if f.endswith(".tif")
        ])
        self.batch_size = batch_size
        self.augment = augment
        self.indexes = np.arange(len(self.naip_files))

        # Sample the data, if specified
        if use_sample:
            sample_size = int(len(self.naip_files) * sample_proportion)

            # Ensure reproducibility
            random.seed(RANDOM_SEED)
            sampled_indices = random.sample(range(len(self.naip_files)), sample_size)

            self.naip_files = [self.naip_files[i] for i in sampled_indices]
            self.chm_files = [self.chm_files[i] for i in sampled_indices]
            self.indexes = np.arange(sample_size)

            print(f"Sampling enabled: using {sample_size} files ({sample_proportion * 100:.1f}%)")

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.naip_files) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data, including augmented versions if applicable."""
        # Compute batch index range
        start = (index % (len(self.naip_files) // self.batch_size)) * self.batch_size
        end = start + self.batch_size
        batch_indexes = self.indexes[start:end]

        # Load batch data
        naip_batch, chm_batch = [], []
        for i in batch_indexes:
            naip = self.load_tif(self.naip_files[i], normalize=True)
            chm = self.load_tif(self.chm_files[i], normalize=False)

            # Apply augmentation with 50% probability
            if self.augment and np.random.rand() < 0.5:
                aug_naip, aug_chm = self.augment_data([naip], [chm])
                naip, chm = aug_naip[0], aug_chm[0]

            naip_batch.append(naip)
            chm_batch.append(chm)

        return np.array(naip_batch), np.array(chm_batch)

    @staticmethod
    def load_tif(file_path, normalize):
        """Loads a single TIF image and optionally normalizes it."""
        with rasterio.open(file_path) as src:
            data = src.read()
            data = np.transpose(data, (1, 2, 0))

        if normalize:
            data = data / 255.0

        return data.astype(np.float32)

    def augment_data(self, naip_batch, chm_batch):
        """Applies random augmentation transforms to input batches."""
        aug_naip, aug_chm = [], []

        for naip, chm in zip(naip_batch, chm_batch):
            naip = tf.convert_to_tensor(naip, dtype=tf.float32)
            chm = tf.convert_to_tensor(chm, dtype=tf.float32)

            # Random flips
            naip = tf.image.random_flip_left_right(naip)
            chm = tf.image.random_flip_left_right(chm)
            naip = tf.image.random_flip_up_down(naip)
            chm = tf.image.random_flip_up_down(chm)

            # Random rotation (±30 degrees)
            angle = np.random.uniform(-30, 30) * np.pi / 180
            naip = tfa.image.rotate(naip, angle)
            chm = tfa.image.rotate(chm, angle)

            # Random zoom (up to 20%)
            zoom = np.random.uniform(1.0, 1.2)
            original_size = tf.shape(naip)[:2]
            new_size = tf.cast(tf.cast(original_size, tf.float32) * zoom, tf.int32)

            naip = tf.image.resize(naip, new_size)
            naip = tf.image.resize_with_crop_or_pad(naip, original_size[0], original_size[1])
            chm = tf.image.resize(chm, new_size)
            chm = tf.image.resize_with_crop_or_pad(chm, original_size[0], original_size[1])

            # Brightness & contrast
            naip = tf.image.random_brightness(naip, max_delta=0.1)
            naip = tf.image.random_contrast(naip, lower=0.9, upper=1.1)

            aug_naip.append(naip.numpy())
            aug_chm.append(chm.numpy())

        return np.array(aug_naip), np.array(aug_chm)


# ------------------------------------------
# UNet Model Architecture
# ------------------------------------------

def unet_model(input_shape=(256, 256, 4)):
    """
    Builds a U-Net model for image-to-image regression.
    Args:
        input_shape (tuple): Shape of the input images.
    Returns:
        model (tf.keras.Model): Compiled U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)

    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)

    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.2)(p4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="linear")(c9)

    return Model(inputs, outputs)

# ------------------------------------------
# Model Training
# ------------------------------------------

def train_model(batch_size, augment, use_sample, sample_proportion, output_name_suffix):

    # Paths
    output_model = os.path.join(model_dir, f"unet_{output_name_suffix}.h5")
    output_csv = output_model.replace(".h5", "_log.csv")

    # Data generators
    trn_gen = DataGenerator(
        naip_dir=os.path.join(data_dir, "train", "naip"),
        chm_dir=os.path.join(data_dir, "train", "chm"),
        batch_size=batch_size,
        augment=augment,
        use_sample=use_sample,
        sample_proportion=sample_proportion
    )
    val_gen = DataGenerator(
        naip_dir=os.path.join(data_dir, "valid", "naip"),
        chm_dir=os.path.join(data_dir, "valid", "chm"),
        batch_size=batch_size,
        augment=augment,
        use_sample=use_sample,
        sample_proportion=sample_proportion
    )

    # Initialize or resume model
    initial_epoch = 0
    if os.path.exists(output_model):
        print(f"Loading existing model from {output_model}")
        model = load_model(output_model, compile=False)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])

        if os.path.exists(output_csv):
            with open(output_csv, "r") as f:
                completed_epochs = len(f.readlines()) - 1
                if completed_epochs == total_epochs:
                    return
                initial_epoch = completed_epochs
                print(f"Resuming from epoch {initial_epoch}")
    else:
        print("Creating new model.")
        model = unet_model()
        model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])

    # Callbacks
    checkpt = ModelCheckpoint(output_model, save_best_only=True, monitor="val_loss", mode="min")
    csvlog = CSVLogger(output_csv, append=True)
    timer = EpochTimer()

    # Train the model
    history = model.fit(
        trn_gen,
        validation_data=val_gen,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[checkpt, csvlog, timer]
    )

# ------------------------------------------
# Main Block
# ------------------------------------------

if __name__ == "__main__":


    # Define parameters
    run_mode = "tune"
    total_epochs = 50
    data_dir = "G:\\pj_cnn_naip\\data"
    model_dir = "S:\\ursa\\campbell\\pj_cnn_naip\\modeling\\cnn_testing"

    # For tuning the hyperpameters
    if run_mode == "tune":

        # Hard code fixed parameters
        sample_proportion = 0.2
        use_sample = True

        # Define variable hyperparameter grid
        batch_sizes = [32]
        augment_flags = [True, False]

        # Loop through grid and train/validate
        for batch_size, augment in itertools.product(batch_sizes, augment_flags):
            suffix = f"tune_ep{total_epochs}_ba{batch_size}_au{int(augment)}"
            print(f"Training with batch_size={batch_size}, augment={augment}")
            train_model(batch_size, augment, use_sample, sample_proportion, suffix)

    # For building the final model
    elif run_mode == "final":

        # Hard-coded best parameters from tuning
        batch_size = 64
        augment = False
        use_sample = False
        sample_proportion = 1.0  # Ignored if use_sample=False
        suffix = f"final_ep{total_epochs}_ba{batch_size}_au{int(augment)}"

        # Train and validate final model
        train_model(batch_size, augment, use_sample, sample_proportion, suffix)
