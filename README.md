# Dog breed classification
Jupyter notebook with experiments on classifying dog breeds from an image and
a Flask app to upload images and run the dog breed predictor

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)

## Installation <a name="installation"></a>

I ran the code on Windows 10 with Anaconda 3.

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/howarthm/DogBreed.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) and [Resnet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.
5. In my Conda environment I installed
  scikit-learn
  scikit-image
  opencv
  matplotlib
  tqdm
  keras
  tensorflow-gpu
6.  Open a terminal and run: jupyter notebook dog_app.ipynb
7.  To run the web app run:  python -m flask run


## Project Motivation<a name="motivation"></a>

For this project, I was wanted to predict a dog breed based on an image.

1. How accurate is the dog breed predictor?
2. What breed would a human be classified as?
3. What model works best?
