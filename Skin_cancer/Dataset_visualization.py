import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


train_dir = r"D:\T2420322 Dataset\ISIC2016\Train\ISBI2016_ISIC_Part3B_Training_Data"                      ######################## for ISIC2016 #########################
test_dir = r"D:\T2420322 Dataset\ISIC2016\Test\ISBI2016_ISIC_Part3B_Test_Data"


def load_and_visualize_images(directory, num_images=5, dataset_type="Dataset"):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Visualizing {dataset_type}: {len(image_files)} images found.")
    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(image_files[:num_images]):  # Limit to first 'num_images'
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis("off")
    plt.show()

#Visualize train and test images
load_and_visualize_images(train_dir, num_images=5, dataset_type="Train Data")
load_and_visualize_images(test_dir, num_images=5, dataset_type="Test Data")

                    ###################################### Not working with the CSV file here ###################################################


#Load data as a TensorFlow dataset      ####################### only train dataset #####################
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode=None,
    image_size=(128, 128),  #Resize images to a consistent size
    batch_size=32  #Number of imagesPbatch
)

# Visualize images
def visualize_tf_images(dataset, num_images=5):
    plt.figure(figsize=(15, 5))
    for images in dataset.take(1):  #first batch
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

visualize_tf_images(train_dataset)
