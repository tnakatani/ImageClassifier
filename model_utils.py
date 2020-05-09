import json
import torch
import logging
from torchvision import models


def set_device(use_cuda=True):
    """Set the device on which a torch.Tensor is allocated to

    Args:
        use_cuda: Boolean whether or not to set device to gpu.
        num_gpus: Number of GPUs to set.
    """
    if torch.cuda.is_available() and use_cuda:
        return torch.device('cuda:0')
    else:
        logging.warn('CUDA is not available. Falling back to CPU')
        return torch.device("cpu")


def select_pretrained_model(model_name: str):
    """Load a pretrained TorchVision model

    Args:
        model_name: Name of the model
    """
    if model_name == 'resnet101':
        return models.resnet101(pretrained=True)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=True)


def freeze_params(model):
    """Freeze parameters so we don't backprop through them"""
    for param in model.parameters():
        param.requires_grad = False


def save_checkpoint(pretrained_name, input_size, output_size, hidden_layers,
                    drop_p, lr, epochs, model, train_data,
                    optimizer, save_dir):
    """Save the trained model for later use"""
    checkpoint = {'pretrained_model': pretrained_name,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'drop_p': drop_p,
                  'learning_rate': lr,
                  'epochs': epochs,
                  'state_dict': model.fc.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'optim_dict': optimizer.state_dict()}
    path = f'{save_dir}/checkpoint.pth'
    torch.save(checkpoint, path)
    print(f'Model saved to {path}')


def map_category_names(cat_to_name:str, classes:list) -> list:
    """Map category labels to category names

    Args:
        cat_to_name: Path to JSON file with category to name mapping
        classes: list of predicted classes

    Returns:
        List of category names
    """
    with open(cat_to_name, 'r') as f:
        cat_dict = json.load(f)
        category_names = []
        for c in classes:
            category_names.append(cat_dict[c])
        return category_names


def print_predictions(category_names:list, probabilities:list):
    """Pretty print prediction results

    Args:
        category_names: List of predicted category names
        probabilities: List of prediction probabilities
    """
    # Header
    h_1, h_2, h_3 = 'Prediction', 'Flower Name', 'Probability'
    print(f'\n{h_1:<12}{h_2:<20}{h_3:<10}')
    # Body
    for i, c in enumerate(category_names):
        pred_i = i+1
        pred_name = c.title()
        pred_prob = f'{probabilities[i]:.2f}'
        print(f'{pred_i:<12}{pred_name:<20}{pred_prob:<10}')


def print_args(args):
    print('Arguments passed:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
