import numpy as np
import os
import cv2
import random

train_dir=input("Enter the path to the training images:")
test_dir=input("Enter the path to the testing images:")
valid_dir=input("Enter the path to the validation images:")

save_loc=input("Enter the path to save the npy files:")

categories=["COVID_19+ve","COVID_19-ve"]
img_size=224


train_images_path=save_loc+"/train_images.npy"
train_labels_path=save_loc+"/train_labels.npy"

test_images_path=save_loc+"/test_images.npy"
test_labels_path=save_loc+"/test_labels.npy"

valid_images_path=save_loc+"/valid_images.npy"
valid_labels_path=save_loc+"/valid_labels.npy"


def generate_file(direc,images_path,labels_path):
    data=[]
    categories=["COVID_19+ve","COVID_19-ve"]

    for category in categories:
        path=os.path.join(direc,category)
        class_num=categories.index(category)
        files=os.listdir(path)
        total=len(files)
        count=1

        for img in files: 
            print("Converting",count,"/", total)              
            try:
                img_array=cv2.imread(os.path.join(path,img))
                img_array=cv2.resize(img_array,(img_size,img_size))
                data.append([img_array,class_num])                
            except Exception as e:
                pass 
            count+=1        

    random.shuffle(data) 
    images=[]
    labels=[]  
    current=1

    for x,y in data:      
      images.append(x)
      labels.append(y)
      current+=1    

    images = np.array(images).reshape(-1,img_size,img_size,3)
    images = images/255.0
    labels = np.array(labels)
    np.save(images_path,images)
    np.save(labels_path,labels)
    print("Files have been saved in the provided directory\n")


generate_file(train_dir,train_images_path,train_labels_path)
generate_file(test_dir,test_images_path,test_labels_path)
generate_file(valid_dir,valid_images_path,valid_labels_path)
