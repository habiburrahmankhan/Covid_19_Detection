# Detecting COVID-19 with Chest X-Ray
As we already know, testing is one of the most important steps towards inhibiting the spread of Covid-19. 
Even though a lot of work has already been done on Covid-19 detection using Chest X-rays, we need a more accurate and efficient model to detect Covid cases.
### we also use GradCam 
---
#### To achieve higher accuracy in the detection of Covid-19 using Chest X-rays, the following model was used: 
###### We used Transfer Learning approach to train the following models:
> Densenet201

> InceptionV3


> Resnet50V2


* An ensemble model, consisting of the three models, is created.
* According to the accuracy yielded by the models individually, weights were assigned to ensemble model.
![flow of program ](https://github.com/habiburrahmankhan/Covid_19_Detection/blob/main/diagram.png)
---
### Data Description 
The dataset used for the project consists:
1. Chest X-ray images of COVID-19 patients: 3600 images
2. Chest X-ray images of other patients and healthy people: 3600 images
* For training, we used 2800 Covid +ve images and 2800 Covid –ve images.
* For validation, we used 400 Covid +ve images and 400 Covid –ve images
* For validation, we used 400 Covid +ve images and 400 Covid –ve images

[The source of the dataset is:](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

---
---
## Dataset Preparation 
* The images are divided into three distinct groups: one for training, one for validation and another for testing.
* All images are resized to 224x224.
* The images as well as their labels are converted into a numpy array.

[link of images as well as their labels are converted into a numpy array ](https://drive.google.com/drive/u/7/folders/1quj5CXy9MmgqJ8kXuxtYP1ElNfvSBpON)
---

---
## Training the model

* All three models (Densenet201, InceptionV3, Resnet50V2) were trained for 30 epochs.
* The value of patience was set to 10 epochs.
* After training all the model save to .h5 

[link of all model in .h5](https://drive.google.com/drive/u/7/folders/1oOtQDZc_uV3hGXJh4bRGF08rd0JcWSTT)
### comparison accuracy of all model 
![Image of accuracy ](https://github.com/habiburrahmankhan/Covid_19_Detection/blob/main/finalmodels_visualization/accuracy_graph.png)

## folder information 
### DataPreparation_Script 
* this file code contsain datapreparation ( image to numpy )
### GUI 
#### A web based GUI was also created.  
#####  all the code contain in GUI folder 
* This will allow anyone to browse a chest X-ray image and upload it to the web application. The application will execute the ensembled model and classify the uploaded Chest X-Ray image.
![front end ](https://github.com/habiburrahmankhan/Covid_19_Detection/blob/main/frontend.png)
### training 
training.py 
> python3 training.py 


testing.py
> python3 testing.py 


test_model.py

### finalmodels_visualization
this folder contain screenshot of  graphs , training time ,  etc. 
---
### A report andd presentation is also available 
### read report or presentation  for clear understanding
* introduction
* literature review 
* proposed method 
* experimental result & explanation 
* conclustion & future work 
---
#### if you want to run on google colab use **covid_run_file.ipynb**
---
## Accuracy of final model: 0.9800

# thank you for reading 
