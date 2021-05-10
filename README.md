
<h1>Singaporean Food Classifier</h1>
<div style="text-align:center"><img src="img/App Screen.png" /></div>
## Table of Contents

1. [Introduction](#introduction)
1. [Model Specifications](#model-specifications)
1. [Model Deployment](#model-deployment)
1. [UI/UX](#ui/ux)
1. [CI/CD Pipeline](#ci/cd-pipeline)
1. [App Limitations and Future Improvement](#app-limitations-and-future-improvement)
1. [Authors](#authors)

## 1.Introduction

This web app aims to help foreigners and people who are not familiar with Singaporean Food to recognise them, together with learning their names and what it consists of. The webpage includes an interactive component asking for an image input from the user, which subsequently returns the most likely food class with the percentage of confidence of the model for the corresponding class. 
<br>
<br>
The webpage also consists of other static information, such as model specifications, detailed description of the 12 classes of food and other useful links.

## 2.Model Specifications

This model is built upon 1224 images of 12 different classes of Singaporean food, with the training,validation and test set split of 60:20:20.
The food classifier is built using transfer learning of the InceptionV3 model, trained using the recognised Imagenet. The pre-trained weights were then used in this model,
of which consisting of 315 layers and 263,820 trainable parameters. The model architecture also consists of batch normalization with early stop monitoring on validation loss, 
helping to prevent overfitting. 
<br>
<br>
The Adaptive Moment Estimation (Adam) optimizer is used for combining the best properties of the AdaGrad and RMSProp algorithms to handle sparse gradients on noisy problems. 
As a result, the model has  a 92.17% validation accuracy over 243 images in the validation set., while also achieving 91.67% accuracy over the test set.

## 3.Model Deployment

### Model Training

This model is built upon 1224 images of 12 different classes of Singaporean food, with the training,validation and test set split of 60:20:20.
The food classifier is built using transfer learning of the InceptionV3 model, trained using the recognised Imagenet. The pre-trained weights were then used in this model,
of which consisting of 315 layers and 263,820 trainable parameters. The model architecture also consists of batch normalization with early stop monitoring on validation loss, 
helping to prevent overfitting. 
<br>
<br>
The model is trained with 20 epochs at the learning rate of 1e-03.

### Input Format

The model accepts several file types, including JPEG, JPG and PNG. There is no restriction on file size and dimension, as dimension will be handled in preprocessing.

### Dataset

The dataset that the model is trained on contains 12 different classes of Singaporean food, including the following:

- 'Chilli Crab'
- 'Curry Puff'
- 'Dim Sum'
- 'Ice Kacang'
- 'Kaya Toast'
- 'Nasi Ayam'
- 'Popiah'
- 'Roti Prata'
- 'Sambal Stingray'
- 'Satay'
- 'Tau Huay'
- 'Wanton Noodle'

The dataset is relatively balanced, with the smallest to the largest class ranging from 60+ to 150+ images. The total dataset size is 1,224 images, of which some contain text.

### Model Performance

The model has  a <b>92.17%</b> validation accuracy over 243 images in the validation set., while also achieving <b>91.67%</b> accuracy over the test set.

## 4.UI/UX

There are several components to the webpage, including 2 html files (index and base), a main javascript file and a css file. There are several functions to improve user experience, such as error messages to user if 'submit' is pressed with no image uploaded. There is also a clear button to reset and delete any uploaded image in the case user wants to start over.
<br>
<br>
In the static information, collapsible boxes are used to not overwhelm user with too much information at the same time, therefore focusing their attention on specific section. In the section '12 Amazing Singaporean Food +', there are nested collapsible boxes for each food for the user to read each individually.

## 5.CI/CD Pipeline

Continuous integration (CI) is the practice of merging all developers' working copies to a shared mainline several times a day. On a high frequency webpage, many features may need to be updated continuously, while multiple developers may work on the same repository at the same time.
<br>
<br>
Continuous deployment (CD) is a software engineering approach in which software functionalities are delivered frequently through automated deployments. CD contrasts with continuous delivery, a similar approach in which software functionalities are also frequently delivered and deemed to be potentially capable of being deployed but are actually not deployed.
<br>
<br>
CI/CD Pipeline refers to a series of steps that must be performed in order to deliver a new version. In a world of fast changes, agile development has become dominant and CI/CD is crucial to the agile approach. CI/CD pipeline introduces monitoring and automation to improve the process of application development, particularly at the integration and testing phases, as well as during delivery and deployment. Although it is possible to manually execute each of the steps of a CI/CD pipeline, the true value of CI/CD pipelines is realized through automation.

## 6.App Limitations and Future Improvement

### Limitations

One of the biggest limitations of the app is that it can only detect Singaporean food, of the 12 classes. If any other images are loaded, it will still return a class out of the 12, which may not be accurate, as it is programmed to return the highest probability. This may be misleading to the user.
<br>
<br>
The model itself is also only trained on 1000+ images, which may not be enough to recognise enough instances of the food. Therefore, the model may not be robust enough when it comes to other images in different lighting or positions. The model is also not able to recognise multiple food in one image. 
<br>
<br>
Another limitation is that there are limited features in this app, therefore its functionality is limited.

### Future Improvement

Future improvement will include improving both the model and the app. [Panoptic Segmentation](https://github.com/topics/panoptic-segmentation) can be used to train a more robust model, together with increasing the classes of food detected and the overall size of the dataset.
<br>
<br>
More features can be added to the application, including a database and cache to save previous uploads, remembering the preferences of the user.

## 7.Authors

This app is created by Harry Tsang. Please contact: [harrytsang92@gmail.com](harrytsang92@gmail.com)