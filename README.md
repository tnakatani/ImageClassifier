# Creating a Flower Image Classifier Using Transfer Learning

The project was completed as part of the [Intro to Machine Learning with PyTorch Nanodegree Program](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229).

This project develops an image classifier trained on a [data set](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of flower images shared by the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/).  The model is constructed using a pretrained model from [`torchvision`](https://pytorch.org/docs/stable/torchvision/models.html) with a feedforward classifier appended to the output stage.

The project was developed in 2 stages:
1. The first stage was developing the idea in a Jupyter notebook, then
2. The second stage was turning the notebook into a command line application.

## Stage 1: Jupyter notebook

The [Image Classifier notebook](notebook/image_classifier.ipynb) works through the steps in developing a neural network.  The notebook is then converted into a Python application that runs from the command line.

## Stage 2: Command Line Application

The functionality of the application is mainly divided into two scripts, `train.py` and `predict.py`.  `train.py` trains a new network on a dataset and save the model as a checkpoint.  The second file, `predict.py`, uses a trained network to predict the class for an input image.

### `train.py`

This script performs the following:

1. Trains a new network on a dataset.  Runs on GPU if available, otherwise falls back to CPU.
2. Prints out training loss, validation loss, and validation accuracy as the network trains.  Additionally stores the output in `train_log.txt`.
3. After the training is complete, saves a checkpoint in the `checkpoint` directory.

Basic usage:

```bash
python train.py --input_dir flowers

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory of dataset for model training
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save checkpoints
  -arch {resnet50,resnet101}
                        Model used for transfer learning
  -input INPUT_SIZE, --input_size INPUT_SIZE
                        Input size of the classifier
  -output OUTPUT_SIZE, --output_size OUTPUT_SIZE
                        Output size of the classifier
  -hidden HIDDEN_LAYERS [HIDDEN_LAYERS ...], --hidden_layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        Number of hidden units per layer;Use like: "-h 1024
                        512 128" for 3 layers of 1024, 512 and 128 hidden
                        units
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate per gradient descent step
  -d DROP_P, --drop_p DROP_P
                        Drop out rate of the classifier during training
  --epochs EPOCHS       Number of training epochs
  --gpu                 Use GPU for training; Default is True
```

`train.py` currently runs the script with the following arguments set in the `consts.py` file:

```python
TRAIN_ARGS = [
    '-i', 'flowers',
    '-o', 'checkpoints',
    '-a', 'resnet101',
    '--input_size', '2048',
    '--output_size', '102',
    '--hidden_layers', '1024', '512',
    '--learning_rate', '0.001',
    '--epochs', '3',
    '--gpu'
]
```

### `predict.py`

This script performs the following:

1. Instantiates a trained model created from the `train.py` script.
2. Reads the image file passed as an argument and predicts the top-K most likely classes.
3. Prints the top-K most likely flower names and the probabilty of each prediction.

Basic usage:

```bash
python predict.py --input_image flowers/test/1/image_06743.jpg

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_IMG, --input_img INPUT_IMG
                        Path to image
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path to checkpoint
  -k TOP_K, --top_k TOP_K
                        Return n most likely classes
  -n CATEGORY_NAMES, --category_names CATEGORY_NAMES
                        Use a mapping of categories to real names
  --gpu                 Use GPU for training; Default is True
```

`predict.py` currently runs the following arguments set in the `consts.py` file:

```python
PREDICT_ARGS = [
    '--input_img', 'flowers/test/1/image_06743.jpg',
    '--checkpoint', 'checkpoints/checkpoint.pth',
    '--top_k', '3',
    '--category_names', 'cat_to_name.json',
    '--gpu'
]
```

An expected output will look like this:

```bash
Arguments passed:
input_img: flowers/test/1/image_06743.jpg
checkpoint: checkpoints/checkpoint.pth
top_k: 3
category_names: cat_to_name.json
gpu: True

Prediction  Flower Name         Probability
1           mexican aster       0.26902204751968384
2           primula             0.18412525951862335
3           hibiscus            0.13510306179523468
```
