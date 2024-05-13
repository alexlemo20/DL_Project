import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_train_val(filepath: str, val_size=0.2):
    data = []
    with open(file=filepath) as f:
        for row in f.readlines():
            row = row.strip().split(' ')
            row[1], row[2], row[3] = int(row[1]), int(row[2]), int(row[3])
            data.append(row)
    df = pd.DataFrame(data=data, columns=['image_id', 'class_id', 'species', 'breed'])
    
    # Replace 1 with 0 (for cats) and 2 with 1 (for dogs)
    df['species'] = df['species'].replace({1: 0, 2: 1})
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df['species'], shuffle=True, random_state=42)
    return train_df, val_df


def load_and_transform_image(image_path):
    """ Load an image and apply the transformations. """
    
    # Define the transformations (we use the same as what Resnet used for efficient transfer)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # Convert all images to RGB
    return transform(image)


# Function to create a tensor dataset from DataFrame and transformations
def create_dataset(df, base_path):
    images_tensors = []
    Y = []
    for row in df.values.tolist():
        image_id, species_id = row[0], row[2]
        image_path = f"{base_path}{image_id}.jpg"  # Adjust format as needed
        image_tensor = load_and_transform_image(image_path)
        images_tensors.append(image_tensor)
        Y.append(species_id)
    
    # Stack all tensors to create a single tensor
    X = torch.stack(images_tensors)
    Y = torch.tensor(np.array(Y))
    return X, Y


def show_images(images, labels, n_images=None, figsize=(8, 8)):
    """
    Display a grid of images with labels.
    """
    if n_images is None:
        n_images = len(images)
    else:
        images = images[:n_images]
        labels = labels[:n_images]
    
    num_rows = int(np.ceil(n_images**0.5))
    num_cols = int(np.ceil(n_images / num_rows))

    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for img, label, ax in zip(images, labels, axes):
        # The image tensor shape should be (channels, height, width)
        img = img.permute(1, 2, 0).numpy()  # Convert to (height, width, channels)
        ax.imshow(img)
        ax.axis('off')  # Hide the axes ticks
        ax.set_title(f"Label: {label.item()}", fontsize=10)  # Set title to the label of the image

    plt.tight_layout()
    plt.show()