import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import argparse
import numpy as np
import json
from PIL import Image
from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('-t', '--top_k', dest='top_k', default='1',
                       help='number of top probabilities - default: 1')
    parser.add_argument('-f', '--filepath', dest='filepath', default=None,
                       help='path to image file for processing')
    parser.add_argument('-c', '--category_names', dest='category_names', default='cat_to_name.json',
                       help='json file with categories/classes to real name mapping')
    parser.add_argument('-g', '--gpu', action='store_true', default=True,
                       help='specify if processing on gpu is preferred')
    return parser.parse_args()

def process_image(image):
    im = Image.open(image)
    thumb_size = 256, 256
    im.thumbnail(thumb_size, Image.ANTIALIAS)
    width = im.width
    height = im.height
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2  
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)/255
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np_image.transpose((2,0,1))

def predict(image_path, model, topk, gpu):
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    model.eval()
    np_image = process_image(image_path)
    image = torch.from_numpy(np.array([np_image])).float()    
    image = image.to(device)
    output = model.forward(image)
    ps = torch.exp(output)

    probs = torch.topk(ps, topk)[0].tolist()[0]
    indices = torch.topk(ps, topk)[1].tolist()[0]
    
    # mapping indices to classes
    class_to_idx_inverted = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = []
    for index in indices:
        classes.append(class_to_idx_inverted[index])

    return probs, classes

def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
        
    if args.filepath:
        img_path = args.filepath
    else:
        print('Cannot run prediction ..')
        img_path = input("Please provide path to image: ")
        
    probs, classes = predict(img_path, model, int(args.top_k), gpu)

    print('\n======')
    print('The filepath of the selected image is: ' + img_path, '\n')    
    print('The top K CLASSES for the selected image are: \n', classes, '\n')
    print('The top K PROBABILITIES for the selected image are: \n ', probs, '\n')   
    print('The top K CATEGORY NAMES for the selected image are: \n', [cat_to_name[x].title() for x in classes])
    print('======\n')

if __name__ == "__main__":
    main()