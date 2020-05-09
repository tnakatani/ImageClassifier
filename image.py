import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def load_data(directory:'str'):
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data, train_loader, valid_loader, test_loader


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    # Resize image to 256x256
    # Image.thumbnail keeps the aspect ratio
    # Note to self: Saving im.thumbnail to a variable results in None (not sure why),
    # thus transform image in place.
    im = Image.open(image)
    width, height = 256, 256
    im.thumbnail((width, height))

    # Crop image by 224x224
    # From PIL documentation (https://pillow.readthedocs.io/en/latest/handbook/concepts.html#coordinate-system):
    # Rectangles are represented as 4-tuples, with the upper left corner given first.
    # For example, a rectangle covering all of an 800x600 pixel image is written as
    # (0, 0, 800, 600)
    resize_w, resize_h = (224, 224)
    left = (width - resize_w)/2
    upper = (height - resize_h)/2
    right = left + resize_w
    lower = upper + resize_h
    im_cropped = im.crop((left, upper, right, lower))

    # Convert values to float 0-1 and normalize
    # Need to divide by 255 to get float: https://knowledge.udacity.com/questions/78361
    np_image = np.array(im_cropped) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_normalized = (np_image - mean) / std

    # Transpose color channel for PyTorch compatibility
    np_transposed = np_normalized.transpose(2, 0, 1)

    # Return PyTorch tensor
    return torch.from_numpy(np_transposed).float()
