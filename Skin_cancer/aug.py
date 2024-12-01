# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import os
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import cv2

# class SkinLesionAugmenter:
#     def __init__(self, csv_path, image_dir, mask_dir, output_img_dir, output_mask_dir, image_size=(256, 256)):
#         """
#         Initialize the augmenter with paths and parameters

#         Args:
#             csv_path: Path to ground truth CSV file
#             image_dir: Directory containing original images
#             mask_dir: Directory containing mask images
#             output_img_dir: Directory to save augmented images
#             output_mask_dir: Directory to save augmented masks
#             image_size: Tuple of (height, width) for resizing images
#         """
#         self.csv_path = csv_path
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.output_img_dir = output_img_dir
#         self.output_mask_dir = output_mask_dir
#         self.image_size = image_size
#         self.df = pd.read_csv(csv_path)

#         # Create output directories
#         os.makedirs(output_img_dir, exist_ok=True)
#         os.makedirs(output_mask_dir, exist_ok=True)

#         # Define augmentation layers
#         self.regular_aug = self._create_augmentation(strength='regular')
#         self.melanoma_aug = self._create_augmentation(strength='strong')

#         # Define augmentation factors based on class distribution
#         self.augment_factors = {
#             'MEL': 2,
#             'NV': 1,
#             'BCC': 10,
#             'AKIEC': 15,
#             'BKL': 1,
#             'DF': 30,
#             'VASC': 25
#         }

#     def _create_augmentation(self, strength='regular'):
#         """
#         Create augmentation pipeline with different strengths

#         Args:
#             strength: 'regular' or 'strong' for different augmentation intensities
#         """
#         # Set parameters based on strength
#         if strength == 'strong':
#             prob = 0.7
#             brightness_range = 0.3
#             zoom_range = 0.2
#             rotation_range = 110
#         else:
#             prob = 0.5
#             brightness_range = 0.2
#             zoom_range = 0.15
#             rotation_range = 90

#         return tf.keras.Sequential([
#             tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
#             tf.keras.layers.RandomRotation(
#                 factor=rotation_range / 360.0,
#                 fill_mode='reflect',
#                 seed=42
#             ),
#             tf.keras.layers.RandomZoom(
#                 height_factor=(-zoom_range, zoom_range),
#                 width_factor=(-zoom_range, zoom_range),
#                 fill_mode='reflect',
#                 seed=42
#             ),
#             tf.keras.layers.RandomBrightness(
#                 factor=brightness_range,
#                 value_range=(0, 255),
#                 seed=42
#             ),
#             tf.keras.layers.RandomContrast(
#                 factor=brightness_range,
#                 seed=42
#             ),
#             # Custom layer for additional augmentations
#             tf.keras.layers.Lambda(lambda x: self._custom_augmentations(x, prob))
#         ])

#     def _custom_augmentations(self, image, prob):
#         """
#         Apply custom augmentations not available in tf.keras.layers
#         """
#         if tf.random.uniform(()) < prob:
#             # Add Gaussian Noise
#             noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
#             image = image + noise
#             image = tf.clip_by_value(image, 0.0, 1.0)

#             # Random Gamma Correction
#             gamma = tf.random.uniform((), 0.8, 1.2)
#             image = tf.pow(image, gamma)

#         return image

#     def preprocess_image(self, image_path):
#         """
#         Load and preprocess image
#         """
#         image = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=self.image_size
#         )
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image = image / 255.0  # Normalize to [0,1]
#         return image

#     def preprocess_mask(self, mask_path):
#         """
#         Load and preprocess mask, ensure it's binary
#         """
#         mask = tf.keras.preprocessing.image.load_img(
#             mask_path,
#             target_size=self.image_size,
#             color_mode='grayscale'  # Ensure the mask is loaded as grayscale
#         )
#         mask = tf.keras.preprocessing.image.img_to_array(mask)
#         mask = mask / 255.0  # Normalize to [0,1]
#         mask = np.where(mask > 0.5, 1, 0)  # Binarize
#         return mask

#     def augment_dataset(self):
#         """
#         Augment the entire dataset
#         """
#         print("Starting dataset augmentation...")

#         # Create tracking variables for augmented data
#         augmented_images = []
#         augmented_labels = []

#         # Process each image in the dataset
#         for idx, row in self.df.iterrows():
#             image_id = row['image']
#             image_path = os.path.join(self.image_dir, image_id + '.jpg')
#             mask_path = os.path.join(self.mask_dir, image_id + '_Segmentation.png')

#             # Debugging statement
#             print(f"Processing image: {image_path}")
#             print(f"Processing mask: {mask_path}")

#             # Skip if image or mask doesn't exist
#             if not os.path.exists(image_path) or not os.path.exists(mask_path):
#                 print(f"Warning: Image or mask {image_path}, {mask_path} not found")
#                 continue

#             # Load and preprocess image and mask
#             image = self.preprocess_image(image_path)
#             mask = self.preprocess_mask(mask_path)

#             # Determine the augmentation factor based on class
#             augment_factor = max(self.augment_factors[col] for col in self.augment_factors if row[col] == 1)
#             augmenter = self.melanoma_aug if row['MEL'] == 1 else self.regular_aug

#             # Generate augmented images and masks
#             for aug_idx in range(augment_factor):
#                 # Apply augmentations
#                 augmented_image = augmenter(tf.expand_dims(image, 0))[0]
#                 augmented_mask = augmenter(tf.expand_dims(mask, 0))[0]

#                 # Ensure the augmented mask is binary
#                 augmented_mask = np.where(augmented_mask > 0.5, 1, 0)

#                 # Generate output filenames
#                 output_img_filename = f"{image_id}_aug_{aug_idx}.jpg"
#                 output_mask_filename = f"{image_id}_aug_{aug_idx}.png"
#                 output_img_path = os.path.join(self.output_img_dir, output_img_filename)
#                 output_mask_path = os.path.join(self.output_mask_dir, output_mask_filename)

#                 # Save augmented image and mask
#                 augmented_image_uint8 = tf.cast(augmented_image * 255, tf.uint8)
#                 augmented_mask_uint8 = tf.cast(augmented_mask * 255, tf.uint8)
#                 tf.keras.preprocessing.image.save_img(
#                     output_img_path,
#                     augmented_image_uint8
#                 )
#                 tf.keras.preprocessing.image.save_img(
#                     output_mask_path,
#                     augmented_mask_uint8,
#                     scale=False
#                 )

#                 # Store labels
#                 labels = [row[col] for col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
#                 augmented_labels.append(labels)

#             if idx % 100 == 0:
#                 print(f"Processed {idx} images...")

#         # Convert labels to binary numpy array
#         augmented_labels = np.array(augmented_labels, dtype=np.float32)

#         return augmented_labels

#     def get_class_distribution(self):
#         """
#         Print the distribution of classes in the original dataset
#         """
#         class_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
#         for col in class_columns:
#             count = self.df[col].sum()
#             print(f"{col}: {count} cases")

# Example usage
# if __name__ == "__main__":
#     augmenter = SkinLesionAugmenter(
#         csv_path='D:\\T2420322 Dataset\\ham10K\\GroundTruth.csv',
#         image_dir='D:\\T2420322 Dataset\\ham10K\\images',
#         mask_dir='D:\\T2420322 Dataset\\ham10K\\masks',
#         output_img_dir='D:\\T2420322 Dataset\\ham10K\\augmented_images',
#         output_mask_dir='D:\\T2420322 Dataset\\ham10K\\augmented_masks',
#         image_size=(256, 256)
#     )

#     # Print original class distribution
#     print("Original class distribution:")
#     augmenter.get_class_distribution()

#     # Perform augmentation
#     augmented_labels = augmenter.augment_dataset()
    
    
    
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

class SkinLesionAugmenter:
    def __init__(self, csv_path, image_dir, mask_dir, output_img_dir, output_mask_dir, image_size=(128, 128)):
        """
        Initialize the augmenter with paths and parameters

        Args:
            csv_path: Path to ground truth CSV file
            image_dir: Directory containing original images
            mask_dir: Directory containing mask images
            output_img_dir: Directory to save augmented images
            output_mask_dir: Directory to save augmented masks
            image_size: Tuple of (height, width) for resizing images
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_img_dir = output_img_dir
        self.output_mask_dir = output_mask_dir
        self.image_size = image_size
        self.df = pd.read_csv(csv_path)

        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        # Define augmentation layers
        self.regular_aug = self._create_augmentation(strength='regular')
        self.melanoma_aug = self._create_augmentation(strength='strong')

        # Define augmentation factors based on class distribution
        self.augment_factors = {
            'MEL': 4,
            'NV': 1,
            'BCC': 4,
            'AKIEC': 9,
            'BKL': 7,
            'DF': 18,
            'VASC': 15
        }

    def _create_augmentation(self, strength='regular'):
        """
        Create augmentation pipeline with different strengths

        Args:
            strength: 'regular' or 'strong' for different augmentation intensities
        """
        # Set parameters based on strength
        if strength == 'strong':
            prob = 0.7
            brightness_range = 0.0015
            zoom_range = 0.05
            rotation_range = 110
            contrast_range= 0.2
        else:
            prob = 0.6
            brightness_range = 0.0015
            zoom_range = 0.06
            rotation_range = 0.12
            contrast_range= 0.12
            
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            tf.keras.layers.RandomRotation(
                factor=rotation_range / 360.0,
                fill_mode='reflect',
                seed=42
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                width_factor=(-zoom_range, zoom_range),
                fill_mode='reflect',
                seed=42
            ),
            tf.keras.layers.RandomBrightness(
                factor=brightness_range,
                value_range=(0, 255),
                seed=42
            ),
            tf.keras.layers.RandomContrast(
                factor=contrast_range,
                seed=42
            ),
            # Custom layer for additional augmentations
            tf.keras.layers.Lambda(lambda x: self._custom_augmentations(x, prob))
        ])

    
    def _custom_augmentations(self, image, prob):
        """
        Apply custom augmentations not available in tf.keras.layers
        """
        if tf.random.uniform(()) < prob:
            # # Apply CLAHE
            # image = self._apply_clahe(image)

            # Random Gamma Correction
            gamma = tf.random.uniform((), 0.9, 1.2)
            image = tf.pow(image, gamma)

        return image

    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        """
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.image_size
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize to [0,1]
        return image

    def preprocess_mask(self, mask_path):
        """
        Load and preprocess mask, ensure it's binary
        """
        mask = tf.keras.preprocessing.image.load_img(
            mask_path,
            target_size=self.image_size,
            color_mode='grayscale'  # Ensure the mask is loaded as grayscale
        )
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255.0  # Normalize to [0,1]
        mask = np.where(mask > 0.5, 1, 0)  # Binarize
        return mask

    def augment_dataset(self):
        """
        Augment the entire dataset
        """
        print("Starting dataset augmentation...")

        # Create tracking variables for augmented data
        augmented_images = []
        augmented_labels = []

        # Process each image in the dataset
        for idx, row in self.df.iterrows():
            image_id = row['image']
            image_path = os.path.join(self.image_dir, image_id + '.jpg')
            mask_path = os.path.join(self.mask_dir, image_id + '_Segmentation.png')

            # Debugging statement
            print(f"Processing image: {image_path}")
            print(f"Processing mask: {mask_path}")

            # Skip if image or mask doesn't exist
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"Warning: Image or mask {image_path}, {mask_path} not found")
                continue

            # Load and preprocess image and mask
            image = self.preprocess_image(image_path)
            mask = self.preprocess_mask(mask_path)

            # Determine the augmentation factor based on class
            augment_factor = max(self.augment_factors[col] for col in self.augment_factors if row[col] == 1)
            augmenter = self.melanoma_aug if row['MEL'] == 1 else self.regular_aug

            # Generate augmented images and masks
            for aug_idx in range(augment_factor):
                # Apply augmentations
                augmented_image = augmenter(tf.expand_dims(image, 0))[0]
                augmented_mask = augmenter(tf.expand_dims(mask, 0))[0]

                # Ensure the augmented mask is binary
                augmented_mask = np.where(augmented_mask > 0.5, 1, 0)

                # Generate output filenames
                output_img_filename = f"{image_id}_aug_{aug_idx}.jpg"
                output_mask_filename = f"{image_id}_aug_{aug_idx}.png"
                output_img_path = os.path.join(self.output_img_dir, output_img_filename)
                output_mask_path = os.path.join(self.output_mask_dir, output_mask_filename)

                # Save augmented image and mask
                augmented_image_uint8 = tf.cast(augmented_image * 255, tf.uint8)
                augmented_mask_uint8 = tf.cast(augmented_mask * 255, tf.uint8)
                tf.keras.preprocessing.image.save_img(
                    output_img_path,
                    augmented_image_uint8
                )
                tf.keras.preprocessing.image.save_img(
                    output_mask_path,
                    augmented_mask_uint8,
                    scale=False
                )

                # Store labels
                labels = [row[col] for col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
                augmented_labels.append(labels)

            if idx % 100 == 0:
                print(f"Processed {idx} images...")

        # Convert labels to binary numpy array
        augmented_labels = np.array(augmented_labels, dtype=np.float32)

        return augmented_labels

    def get_class_distribution(self):
        """
        Print the distribution of classes in the original dataset
        """
        class_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        for col in class_columns:
            count = self.df[col].sum()
            print(f"{col}: {count} cases")

# Example usage
if __name__ == "__main__":
    augmenter = SkinLesionAugmenter(
        csv_path='D:\\T2420322 Dataset\\ham10K\\GroundTruth.csv',
        image_dir='D:\\T2420322 Dataset\\ham10K\\images',
        mask_dir='D:\\T2420322 Dataset\\ham10K\\masks',
        output_img_dir='D:\\T2420322 Dataset\\ham10K\\augmented_images',
        output_mask_dir='D:\\T2420322 Dataset\\ham10K\\augmented_masks',
        image_size=(256, 256)
    )

    # Print original class distribution
    print("Original class distribution:")
    augmenter.get_class_distribution()

    # Perform augmentation
    augmented_labels = augmenter.augment_dataset()
