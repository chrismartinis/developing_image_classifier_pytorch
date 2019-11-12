# Part 1 - Developing an Image Classifier with Deep Learning

Project Intro - _[Youtube](https://www.youtube.com/watch?v=--9IFCNBM6Y)_

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please _[refer to the rubric](https://review.udacity.com/#!/rubrics/1663/view)_ for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize.

This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files on GitHub _[here](https://github.com/udacity/aipnd-project)_.

# Part 2 - Building the command line application

## Specifications
The project submission must include at least two files ```train.py``` and ```predict.py```. The first file, ```train.py```, will train a new network on a dataset and save the model as a checkpoint. The second file, ```predict.py```, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. __Make sure to include all files necessary to run ```train.py``` and ```predict.py``` in your submission__.

- Train a new network on a data set with ```train.py```

  - Basic usage: ```python train.py data_directory```
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    - Choose architecture: ```python train.py data_dir --arch "vgg13"```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
    - Use GPU for training: ```python train.py data_dir --gpu```
- Predict flower name from an image with ```predict.py``` along with the probability of that name. That is, you'll pass in a single image ```/path/to/image``` and return the flower name and class probability.

  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint --top_k 3```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```
