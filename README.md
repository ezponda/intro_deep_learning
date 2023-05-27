# Practical Deep Learning Course

Welcome to the GitHub repository for the Practical Deep Learning Course. Here you will find all the code, notebooks, and resources used throughout the course.


## Repository Content

The content in this repository is divided into five key modules, focusing on various aspects of **Deep Learning**. Each module includes Jupyter notebooks combining theory, code, and explanations.

- **Module 1: Deep Learning Fundamentals**: 'class/Fundamentals'
  - [NN_Fundamentals](./class/Fundamentals/NN_Fundamentals.ipynb): This notebook introduces the basics of Deep Learning using TensorFlow. You will learn about TensorFlow's Sequential and Functional API, which are key tools for building neural networks.
  - [Prevent_Overfitting](./class/Fundamentals/Prevent_Overfitting.ipynb): Here we explore various Regularization Techniques that can prevent overfitting in neural networks.

- **Module 2: Convolutional Neural Networks (CNNs)**: 'class/CNN'
  - [Introduction_to_CNN.ipynb](./class/CNN/Introduction_to_CNN.ipynb): This notebook provides an introduction to Convolutional Neural Networks (CNNs) and their applications in Image Processing.

- **Module 3: Recurrent Neural Networks (RNNs)**: 'class/RNN'
  - [Introduction_to_RNN_Time_Series.ipynb](./class/RNN/Introduction_to_RNN_Time_Series.ipynb): Here we introduce Recurrent Neural Networks (RNNs) and show how they can be used for Time Series Analysis.

- **Module 4: Natural Language Processing (NLP) with Deep Learning**: 'class/NLP'

- **Module 5: Generative Models**: 'class/generative'

In addition to these practical modules, we provide a brief theoretical overview in the [Deep Learning with TensorFlow notebook](./Deep_learning_with_Tensorflow.ipynb). This resource offers a concise recap of essential deep learning concepts, acting as a handy reference guide.

## How to Run the Code

The Jupyter notebooks provided in this course can be run either locally on your machine or remotely on [Google Colab](https://colab.research.google.com/)

### Running Locally

To install TensorFlow and TensorFlow-hub, run:
```shell
pip install tensorflow
pip install --upgrade tensorflow-hub
```
For more details see [TensorFlow installation instructions](https://www.tensorflow.org/install)

To install all the dependencies, run:
```shell
pip install -r requirements.txt
```

### Running on Google Colab

You can upload the notebook directly in [Google Colab](https://colab.research.google.com/) or you can click in the Colab icon at the beginning of each notebook:

<table align="center">
 <td align="center">
        <img src="https://i.ibb.co/2P3SLwK/colab.png"  style="padding-bottom:5px;" />Run in Google Colab</a></td>
  <td align="center">
        <img src="https://i.ibb.co/xfJbPmL/github.png"  height="70px" style="padding-bottom:5px;"  />View Source on GitHub</a></td>
</table>

### Running with Docker
In addition to running the notebooks locally or on Google Colab, you can also run them in a [Docker]((https://www.docker.com/)) container. This ensures that all dependencies are satisfied in a self-contained environment, which can make it easier to get up and running with the project.

Follow these steps to build and run the Docker image:


#### Pre-requisites

Install Docker on your machine. You can download it for Mac, Windows, or Linux from the [official Docker website](https://www.docker.com/products/docker-desktop).


#### Building the Docker Image

Open your terminal, navigate to the directory containing the Dockerfile and run the following command to build the Docker image:

```shell
docker build -t deep-learning-course .
```
This command builds an image and tags it as "deep-learning-course".


#### Running the Docker Image

Run the Docker image with the following command:

```shell
docker run -p 4000:8888 deep-learning-course
```

To persist your changes and have your notebooks saved outside of Docker (on Windows, you might have to replace
`$(pwd)` with `${pwd}` or with the full path to your directory):

```shell
docker run -p 4000:8888 -v "$(pwd)":/app deep-learning-course
```


This command maps the port 8888 inside Docker as port 4000 on your machine.


#### Accessing the Notebooks

Once your Docker image is up and running, you can access the Jupyter notebooks by visiting [http://localhost:4000](http://localhost:4000) in your web browser. You'll see a list of notebooks that you can click on to view, run, and interact with.
    