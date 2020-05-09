import torch
from torchvision import datasets, transforms

def load_data(directory: 'str'):
    """Load image data using torchvision, apply transforms and
    split into training, validation, and testing DataLoaders.

    Args:
        directory: Directory of the training, validation and
        testing data.
    """

    # Set directory paths
    train_dir = directory + '/train'
    valid_dir = directory + '/valid'
    test_dir = directory + '/test'

    # Set transform constants
    normalize = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize = 255
    crop = 224

    # Training
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25,
                               contrast=0.25,
                               saturation=0.25),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(normalize, std)])

    # Validation
    valid_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(normalize, std)])

    # Testing
    test_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(normalize, std)])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data, train_loader, valid_loader, test_loader
