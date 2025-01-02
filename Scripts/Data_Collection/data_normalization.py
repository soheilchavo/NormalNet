import torch
import torchvision.datasets as datasets
from torchvision import transforms

batch_size = 16

#Calculates the mean and standard deviation of a dataset
def calculate_dataset_mean_and_std(dataset_loader : torch.utils.data.DataLoader):
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in dataset_loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()
    mean /= num_pixels
    std /= num_pixels
    return mean, std

#Takes a dataset as a folder of images and normalizes it
def normalize_data(dataset_path : str):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=data_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mean, std = calculate_dataset_mean_and_std(loader)

    data_transforms = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=data_transforms)
    return dataset, loader
