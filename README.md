# Practical Deep Learning Course

Welcome to the GitHub repository for the Practical Deep Learning Course. Here you will find all the code, notebooks, and resources used throughout the course.

## Table of Contents

- [Repository Content](#repository-content)
- [How to Run the Code](#how-to-run-the-code)


## Repository Content

The content in this repository is divided into five key modules, focusing on various aspects of **Deep Learning**. Each module includes Jupyter notebooks combining theory, code, and explanations.

- **Module 1: Deep Learning Fundamentals**: [`class/Fundamentals`](./class/Fundamentals)
  - [First_Model](./class/Fundamentals/First_Model.ipynb): First neural network — MNIST handwritten digit classification.
  - [NN_Fundamentals](./class/Fundamentals/NN_Fundamentals.ipynb): Neural network basics with decision boundary visualization on toy datasets.
  - [IMBD_sentiment_binary_classification](./class/Fundamentals/IMBD_sentiment_binary_classification.ipynb): Binary sentiment classification on movie reviews.
  - [Prevent_Overfitting](./class/Fundamentals/Prevent_Overfitting.ipynb): Regularization techniques (Dropout, L1/L2, Early Stopping) to prevent overfitting.
  - [Regression_tuner](./class/Fundamentals/Regression_tuner.ipynb): Regression with hyperparameter optimization using Keras Tuner.

- **Module 2: Convolutional Neural Networks (CNNs)**: [`class/CNN`](./class/CNN)
  - [Introduction_to_CV_with_Pillow](./class/CNN/Introduction_to_CV_with_Pillow.ipynb): Image manipulation and processing with Pillow.
  - [Introduction_to_CV_with_OpenCV](./class/CNN/Introduction_to_CV_with_OpenCV.ipynb): Computer vision basics using OpenCV.
  - [Introduction_to_CNN](./class/CNN/Introduction_to_CNN.ipynb): Convolution operations, CNN architectures, transfer learning and data augmentation.
  - [Classical_Architectures_CNN](./class/CNN/Classical_Architectures_CNN.ipynb): Classic CNN architectures (LeNet, AlexNet, VGG, ResNet, Inception).
  - [cat_vs_dogs](./class/CNN/cat_vs_dogs.ipynb): Binary image classification with hyperparameter tuning.
  - [Visualizing_What_CNNs_Learn](./class/CNN/Visualizing_What_CNNs_Learn.ipynb): Feature maps, Grad-CAM and saliency maps.
  - [OCR_with_OpenCV_and_Tesseract](./class/CNN/OCR_with_OpenCV_and_Tesseract.ipynb): Optical Character Recognition with Tesseract.
  - [Object_Detection_YOLO_ultralytics](./class/CNN/Object_Detection_YOLO_ultralytics.ipynb): Object detection with YOLO and Ultralytics.
  - [Object_Tracking_Counting](./class/CNN/Object_Tracking_Counting.ipynb): Object tracking and counting with YOLO.
  - [Siamese_net](./class/CNN/Siamese_net.ipynb): Siamese networks for similarity learning.

- **Module 3: Recurrent Neural Networks (RNNs)**: [`class/RNN`](./class/RNN)
  - [Introduction_to_RNN_Time_Series](./class/RNN/Introduction_to_RNN_Time_Series.ipynb): RNN fundamentals and time series prediction.
  - [IMBD_RNN](./class/RNN/IMBD_RNN.ipynb): Sentiment classification using RNNs.
  - [Character-level_text_generation_with_RNN](./class/RNN/Character-level_text_generation_with_RNN.ipynb): Character-level text generation.
  - [Seq2seq](./class/RNN/Seq2seq.ipynb): Sequence-to-sequence models.
  - [img2seq](./class/RNN/img2seq.ipynb): Image-to-sequence model (CAPTCHA solving).

- **Module 4: Natural Language Processing (NLP) with Deep Learning**: [`class/NLP`](./class/NLP)
  - [Embedding_layer](./class/NLP/Embedding_layer.ipynb): Word embeddings for text representation.
  - [text_classification_rnn](./class/NLP/text_classification_rnn.ipynb): Text classification using RNNs and CNNs.
  - [semantic_search_QA_clustering](./class/NLP/semantic_search_QA_clustering.ipynb): Semantic search and question-answering with Sentence Transformers.
  - [Image_search](./class/NLP/Image_search.ipynb): Image search using CLIP and Sentence Transformers.

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
 <td align="center"><a target="_blank" ></a>
        <img src="https://colab.research.google.com/img/colab_favicon_256px.png"  width="50" height="50" style="padding-bottom:5px;" />Run in Google Colab</a></td>
  <td align="center"><a target="_blank" ></a>
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"  width="50" height="50" style="padding-bottom:5px;" />View Source on GitHub</a></td>
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
    