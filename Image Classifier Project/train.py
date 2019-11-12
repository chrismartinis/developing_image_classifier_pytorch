import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import argparse
from utils import save_checkpoint, load_checkpoint
from workspace_utils import active_session

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-d', '--data_dir', action='store',
                        help='data directory where images are stored')
    parser.add_argument('-a', '--arch', dest='arch', default='densenet121', choices=['densenet121', 'densenet169'], 
                       help='model architexture to use')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', default='0.001',
                       help='learning rate for gradient descent')
    parser.add_argument('-u', '--hidden_units', dest='hidden_units', default='512',
                       help='number of hidden units the classifier will be using')
    parser.add_argument('-e', '--epochs', dest='epochs', default='5',
                       help='number of epochs the training model will loop through')
    parser.add_argument('-g', '--gpu', action="store_true", default=True,
                       help='specify if processing on gpu is preferred')
    return parser.parse_args()

def validation(model, dataloaders, criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    valid_loss = 0
    accuracy = 0
    for images, labels in dataloaders[1]:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    epochs = epochs
    steps = 0
    print_every = 40
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    print('Training has started ..')
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in dataloaders[0]:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloaders, criterion)

                print('Epoch: {}/{}..'.format(e+1, epochs),
                     'Training Loss: {:.3f}..'.format(running_loss/print_every),
                     'Validation Loss: {:.3f}..'.format(valid_loss/len(dataloaders[1])),
                      'Accuracy: {:.3f}'.format(accuracy/len(dataloaders[1])))
                running_loss = 0
                model.train()
    print('==Training has ended==')

def main():
    args = parse_args()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
#     test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])
    
#     test_transforms = valid_transforms
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
#     test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
#     testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    dataloaders = [trainloader, validloader]
    
    model = getattr(torchvision.models, args.arch)(pretrained=True)
    for param in model.parameters():
        param_requires_grad = False
        
    if args.arch == "densenet121":
        features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(features, int(args.hidden_units))),
                                  ('dropout', nn.Dropout(p=0.4)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(int(args.hidden_units), 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet169":
        features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(features, int(args.hidden_units))),
                                  ('dropout', nn.Dropout(p=0.4)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(int(args.hidden_units), 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
            
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
        
    epochs = int(args.epochs)
    gpu = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    with active_session():
        train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, optimizer, args, classifier)

if __name__ == "__main__":
    main()
