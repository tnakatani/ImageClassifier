"""Predict a flower name from an image using a trained model.
Returns the flower name and class probability.

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
from image import process_image
from model_utils import select_pretrained_model, freeze_params, map_category_names, print_predictions, print_args
from network import Network

def init_argparse(*args):
    """Instantiate argparse object"""
    parser = argparse.ArgumentParser(
        description='Train a network on dataset and save the model as a checkpoint'
    )
    parser.add_argument('-i', '--input_img',
                        help='Path to image')
    parser.add_argument('-c', '--checkpoint',
                        help='Path to checkpoint',
                        default='checkpoints')
    parser.add_argument('-k', '--top_k',
                        help='Return n most likely classes',
                        type=int,
                        default=3)
    parser.add_argument('-n', '--category_names',
                        help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', help='Use GPU for training; Default is True',
                       action='store_true',
                       default=True)
    # Initialize with constants if passed in as an argument
    if args:
        return parser.parse_args(args[0])
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path)

    # Load pretrained model
    model = select_pretrained_model(checkpoint['pretrained_model'])

    # Freeze parameters to prevent backpropagation
    freeze_params(model)

    # Load classifier
    classifier = Network(checkpoint['input_size'],
                 checkpoint['output_size'],
                 checkpoint['hidden_layers'],
                 checkpoint['drop_p'])
    classifier.load_state_dict(checkpoint['state_dict'])

    # Merge classifier to end of pretrained model
    model.fc = classifier

    # Add class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    # Invert class_to_idx dictionary
    # Ref: https://therenegadecoder.com/code/how-to-invert-a-dictionary-in-python/#invert-a-dictionary-with-a-comprehension
    model.idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}

    return model


def predict(image_path, model, k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Disable dropout
    model.eval()

    # Disable autograd
    with torch.no_grad():
        # Process image to PyTorch tensor
        img = process_image(image_path)
        # Need to unsqueeze for a single image
        # Ref: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
        img.unsqueeze_(0)

        # Hack to get around "RunTimeError Expected object of type
        # torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'"
        # Ref: https://knowledge.udacity.com/questions/131608
        model.to('cpu')
        img.to('cpu')

        # Get probability distribution
        output = model(img)
        ps = torch.exp(output)

        # Get top k probabilities and classes
        top_p, top_classes = ps.topk(k, dim=1)

        # Convert top_p, top_classes tensors to plain lists for easier
        # ingestion downstream.
        # Ref: https://stackoverflow.com/a/53903817
        probs = top_p.squeeze().tolist()
        classes = [model.idx_to_class[i] for i in top_classes.squeeze().tolist()]

        logging.info(f'Probability distribution: {ps}')
        logging.info(probs)
        logging.info(classes)

        return probs, classes


if __name__ == '__main__':
    logging.basicConfig(filename='predict_log.txt', level=logging.INFO)
    args = init_argparse(consts.PREDICT_ARGS)
    print_args(args)
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(image_path=args.input_img, model=model, k=args.top_k)
    pred_labels = map_category_names(cat_to_name=args.category_names,
                                     classes=classes)
    print_predictions(pred_labels, probs)

