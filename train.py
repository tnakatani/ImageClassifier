"""Train a network on dataset and save the model as a checkpoint

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_layers 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
"""

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import logging
import json
import argparse
import consts
from image import load_data
from model_utils import set_device, select_pretrained_model, freeze_params, save_checkpoint, print_args
from network import Network


def init_argparse(*args):
    """Instantiate argparse object"""
    parser = argparse.ArgumentParser(
        description='Train a network on dataset and save the model as a checkpoint'
    )
    parser.add_argument('-i', '--input_dir',
                        help='Directory of dataset for model training')
    parser.add_argument('-o', '--output_dir',
                        help='Directory to save checkpoints',
                        default='checkpoints')
    parser.add_argument('-arch',
                        help='Model used for transfer learning',
                        choices=['resnet50', 'resnet101'])
    parser.add_argument('-input', '--input_size',
                        help='Input size of the classifier',
                        type=int)
    parser.add_argument('-output', '--output_size',
                        help='Output size of the classifier',
                        type=int)
    parser.add_argument('-hidden', '--hidden_layers',
                        help='Number of hidden units per layer;' +
                             'Use like: "-h 1024 512 128" for 3 layers of 1024, 512 and 128 hidden units',
                        type=int,
                        nargs='+')
    parser.add_argument('-l', '--learning_rate',
                        help='Learning rate per gradient descent step',
                        type=float,
                        default=0.001)
    parser.add_argument('-d', '--drop_p',
                        help='Drop out rate of the classifier during training',
                        type=float,
                        default=0.25)
    parser.add_argument('--epochs',
                        help='Number of training epochs',
                        type=int,
                        default=1)
    parser.add_argument('--gpu', help='Use GPU for training; Default is True',
                       action='store_true',
                       default=True)
    # Initialize with constants if passed in as an argument
    if args:
        return parser.parse_args(args[0])
    return parser.parse_args()


def define_classifier(input_size: int, output_size: int,
                      hidden_layers: list, drop_p: float):
    """Instantiate a classifier class

    Args:
        input_size: Number of input features
        output_size: Number of output features
        hidden_layers: List of hidden units per layer
        drop_p: Drop out rate
    """
    return Network(input_size=input_size,
                   output_size=output_size,
                   hidden_layers=hidden_layers,
                   drop_p=drop_p)

def define_network(pretrained_model, classifier, learning_rate: float,
                   epochs: int):
    """Load pre-trained network and append a custom classifier.

    Args:
        pretrained_model: pretrained model
        classifier: classifier appended to the pretrained model
        learning_rate: rate of gradient descent steps
        epoch: number of forward and backwards pass conducted

    """

    # Merge pretrained model and freeze parameters
    # to prevent backpropagation
    model = select_pretrained_model(pretrained_model)
    freeze_params(model)

    # freeze_params() must come before setting classifier, otherwise you get:
    # ValueError: optimizing a parameter that doesn't require gradients
    model.fc = classifier

    # Define loss and optimization
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_network(model, criterion, optimizer, epochs, cuda):

    # Set logging parameters
    steps = 0
    print_every = 5
    running_loss = 0

    # Use CUDA if available
    device = set_device(cuda)
    model.to(device)

    # keep task awake
    with active_session():
        for e in range(epochs):
            for images, labels in train_loader:
                steps += 1

                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)

                # Reset optimization
                optimizer.zero_grad()

                # Forwardpass, backprop, optimize
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                # Increment loss
                running_loss += loss.item()

                # Print metrics every n steps
                if steps % print_every == 0:
                    # Disable dropout for evaluation
                    model.eval()

                    # Initialize validation metrics
                    valid_loss = 0
                    accuracy = 0

                    # Start evaluation loop

                    # Disable autograd to speed up evaluation
                    with torch.no_grad():
                        for images, labels in valid_loader:
                            # Move input and label tensors to the default device
                            images, labels = images.to(device), labels.to(device)

                            # Pass images into model to get log probabilities, and
                            # increment loss on our test set as it is training
                            log_ps = model.forward(images)
                            loss = criterion(log_ps, labels)
                            valid_loss += loss.item()

                            # Get probability distribution, top predicted class from
                            # the network, check where our prediction match labels and
                            # increment accuracy metrics.
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Report and Log Metrics
                    msg = (f"Epoch {e+1}/{epochs}.. "
                           f"Train loss: {running_loss/print_every:.3f}.. "
                           f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                           f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                    print(msg)
                    logging.info(msg)

                    # Reset loss and set model back to training mode
                    running_loss = 0
                    model.train()
    print('Training complete')


if __name__ == '__main__':
    logging.basicConfig(filename='train_log.txt', level=logging.INFO)
    args = init_argparse(consts.TRAIN_ARGS)
    print_args(args)
    train_data, train_loader, valid_loader, test_loader = load_data(args.input_dir)
    classifier = define_classifier(args.input_size, args.output_size,
                                   args.hidden_layers, args.drop_p)
    model, criterion, optimizer = define_network(args.arch, classifier,
                                                 args.learning_rate, args.epochs)
    train_network(model, criterion, optimizer, args.epochs, args.gpu)
    save_checkpoint(args.arch, args.input_size, args.output_size,
                    args.hidden_layers, args.drop_p,
                    args.learning_rate, args.epochs,
                    model, train_data, optimizer, args.output_dir)
