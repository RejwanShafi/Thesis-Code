import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'D:/T2420322 Dataset/ISIC2016/Train/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
data = pd.read_csv(file_path)

# Check the contents of the dataset
print(data.head())

# Count the distinct cases of skin lesions if present in a column (assuming the dataset has a column for lesion type)
if 'lesion_type' in data.columns:
    # Count occurrences of each type of lesion
    lesion_counts = data['lesion_type'].value_counts()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=lesion_counts.index, y=lesion_counts.values, palette='viridis')
    plt.title('Number of Distinct Cases of Skin Lesions in ISIC 2016 Dataset')
    plt.xlabel('Lesion Type')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("The dataset does not have a 'lesion_type' column. Please check your dataset structure.")
