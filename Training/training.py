import numpy as np
from train_model import *
from tensorflow.keras.callbacks import EarlyStopping


classes=["COVID_19 +ve","COVID_19 -ve"]
img_size=224
no_of_epochs=30


train_images_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/train_images.npy"
train_labels_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/train_labels.npy"


valid_images_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/valid_images.npy"
valid_labels_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/valid_labels.npy"

model_path="/content/gdrive/MyDrive/minor_project_new/finalmodels/"


train_images=np.load(train_images_path)
train_labels=np.load(train_labels_path)


valid_images=np.load(valid_images_path)
valid_labels=np.load(valid_labels_path)


early_stop=EarlyStopping(monitor='val_loss',patience=10,verbose=0, 
                         mode='min',restore_best_weights=True)


densenet=train_model(model_path,train_images,train_labels,
                     valid_images,valid_labels,model_name="densenet201",
                     epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                     classes=len(classes),callbacks=[early_stop])


inception=train_model(model_path,train_images,train_labels,
                      valid_images,valid_labels,model_name="inception_v3",
                      epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                      classes=len(classes),callbacks=[early_stop])


resnet=train_model(model_path,train_images,train_labels,
                   valid_images,valid_labels,model_name="resnet50_v2",
                   epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                   classes=len(classes),callbacks=[early_stop])

