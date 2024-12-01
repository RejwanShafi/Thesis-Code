import os
import pandas as pd

class AugmentedCSVGenerator:
    def __init__(self, original_csv_path, output_csv_path, image_dir, mask_dir):
        """
        Initialize the generator with paths and parameters.
        
        Args:
            original_csv_path: Path to the original CSV file.
            output_csv_path: Path where the new CSV file will be saved.
            image_dir: Directory containing the original and augmented images.
            mask_dir: Directory containing the original and augmented masks.
        """
        self.original_csv_path = original_csv_path
        self.output_csv_path = output_csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def generate_csv(self):
        """
        Generate a new CSV file containing the paths to images, masks, and their labels.
        """
        # Read the original CSV
        original_df = pd.read_csv(self.original_csv_path)

        # Prepare a new dataframe to hold all entries
        augmented_entries = []

        # Process each entry in the original dataset
        for _, row in original_df.iterrows():
            image_id = row['image']
            image_path = os.path.join(self.image_dir, image_id + '.jpg')
            mask_path = os.path.join(self.mask_dir, image_id + '_Segmentation.png')

            # Add the original image and mask to the new dataset
            augmented_entries.append({
                'image_path': image_path,
                'mask_path': mask_path,
                **{col: row[col] for col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']}
            })

            # Look for augmented images for this image_id
            aug_idx = 0
            while True:
                augmented_image_path = os.path.join(self.image_dir, f"{image_id}_aug_{aug_idx}.jpg")
                augmented_mask_path = os.path.join(self.mask_dir, f"{image_id}_aug_{aug_idx}.png")

                # If augmented files exist, add them to the new dataset
                if os.path.exists(augmented_image_path) and os.path.exists(augmented_mask_path):
                    augmented_entries.append({
                        'image_path': augmented_image_path,
                        'mask_path': augmented_mask_path,
                        **{col: row[col] for col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']}
                    })
                    aug_idx += 1
                else:
                    break

        # Convert the collected entries into a DataFrame
        augmented_df = pd.DataFrame(augmented_entries)

        # Save the new DataFrame as a CSV file
        augmented_df.to_csv(self.output_csv_path, index=False)
        print(f"Augmented dataset CSV saved at {self.output_csv_path}")


def count_lesion_types(augmented_csv_path):
    """
    Count the number of cases for each lesion type in the augmented dataset.

    Args:
        augmented_csv_path: Path to the augmented CSV file.

    Returns:
        A dictionary with lesion types as keys and their counts as values.
    """
    # Read the augmented CSV
    augmented_df = pd.read_csv(augmented_csv_path)
    
    # List of lesion type columns
    lesion_types = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    # Count the number of occurrences for each lesion type
    counts = {lesion_type: augmented_df[lesion_type].sum() for lesion_type in lesion_types}
    
    return counts


# Example usage
if __name__ == "__main__":
    # Paths to the necessary files and directories
    original_csv_path = 'D:\\T2420322 Dataset\\ham10K\\GroundTruth.csv'
    output_csv_path = 'D:\\T2420322 Dataset\\ham10K\\augmented_dataset.csv'
    image_dir = 'D:\\T2420322 Dataset\\ham10K\\augmented_images'
    mask_dir = 'D:\\T2420322 Dataset\\ham10K\\augmented_masks'

    # Generating the augmented CSV
    generator = AugmentedCSVGenerator(
        original_csv_path=original_csv_path,
        output_csv_path=output_csv_path,
        image_dir=image_dir,
        mask_dir=mask_dir
    )
    generator.generate_csv()

    # Count lesion types in the augmented CSV
    lesion_counts = count_lesion_types(output_csv_path)
    print("Lesion type counts:")
    for lesion_type, count in lesion_counts.items():
        print(f"{lesion_type}: {count}")
