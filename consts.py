# argparse
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

PREDICT_ARGS = [
    '--input_img', 'flowers/test/1/image_06743.jpg',
    '--checkpoint', 'checkpoints/checkpoint.pth',
    '--top_k', '3',
    '--category_names', 'cat_to_name.json',
    '--gpu'
]

# torchvision transforms
NORMALIZE = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE = 255
CROP = 224
