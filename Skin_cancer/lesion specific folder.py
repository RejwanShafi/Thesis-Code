import os
import pandas as pd
import shutil

# Define paths
csv_file_path = 'D:\\T2420322 Dataset\\ham10K\\GroundTruth.csv'
images_folder = 'D:\\T2420322 Dataset\\ham10K\\images'
masks_folder = 'D:\\T2420322 Dataset\\ham10K\\masks'
output_folder = 'D:\\T2420322 Dataset\\ham10K\\organized_lesions'

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Iterate through the CSV rows
for index, row in data.iterrows():
    image_name = row['image']  # Column for the image filename without extension

    # Add the jpg extension to the image name
    image_filename = f"{image_name}.jpg"
    mask_filename = f"{image_name}-segmentation.png"

    # Determine the lesion type based on the one-hot encoded columns
    lesion_types = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    lesion_type = None
    for lesion in lesion_types:
        if row[lesion] == 1:
            lesion_type = lesion
            break

    # Skip if no lesion type is identified
    if lesion_type is None:
        print(f"No lesion type found for image: {image_name}")
        continue

    # Create lesion-specific folder structure
    lesion_folder = os.path.join(output_folder, lesion_type)
    lesion_images_folder = os.path.join(lesion_folder, 'images')
    lesion_masks_folder = os.path.join(lesion_folder, 'masks')

    os.makedirs(lesion_images_folder, exist_ok=True)
    os.makedirs(lesion_masks_folder, exist_ok=True)

    # Paths for image and mask
    image_path = os.path.join(images_folder, image_filename)
    mask_path = os.path.join(masks_folder, mask_filename)

    # Copy image and mask to the respective folders
    if os.path.exists(image_path):
        shutil.copy(image_path, lesion_images_folder)
    else:
        print(f"Image not found: {image_path}")

    if os.path.exists(mask_path):
        shutil.copy(mask_path, lesion_masks_folder)
    else:
        print(f"Mask not found: {mask_path}")

print("Organizing completed.")