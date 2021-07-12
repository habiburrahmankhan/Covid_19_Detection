import tensorflow as tf 
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from progressbar import ProgressBar


img_size=224
classes=["COVID_19 +ve","COVID_19 -ve"]


test_images_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/test_images.npy"
test_labels_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/test_labels.npy"

valid_images_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/valid_images.npy"
valid_labels_path="/content/gdrive/MyDrive/minor_project_new/train_test_val_npy_file/valid_labels.npy"


densenet_path="/content/gdrive/MyDrive/minor_project_new/models1/train_test_val_npy_file/densenet201.h5"
inception_path="/content/gdrive/MyDrive/minor_project_new/models2/train_test_val_npy_file/inception_v3.h5"
resnet_path="/content/gdrive/MyDrive/minor_project_new/models3/train_test_val_npy_file/resnet50_v2.h5"

test_images=np.load(test_images_path)
test_labels=np.load(test_labels_path)

valid_images=np.load(valid_images_path)
valid_labels=np.load(valid_labels_path)

densenet_model=tf.keras.models.load_model(densenet_path)
inception_model=tf.keras.models.load_model(inception_path)
resnet_model=tf.keras.models.load_model(resnet_path)

models=[densenet_model,inception_model,resnet_model]



def get_weights(test_images,test_labels,models):
    
    accuracy=[]
    weights=np.full((1,len(models)),100.0)
    for model in models:
      model.compile(optimizer='adam',loss='mean_squared_error', 
                    metrics=['sparse_categorical_accuracy'])
      acc=model.evaluate(test_images,test_labels)[1]
      accuracy.append(100*acc)
    weights=weights-accuracy
    weights=weights**2
    tot=np.sum(weights)
    weights=weights/tot
    weights=1/weights
    weights=weights**2
    tot=np.sum(weights)
    weights=weights/tot
    return weights

w=get_weights(test_images,test_labels,models)[0]
print("Weights: ",w)
#w[0]=1
#w[1]=1
#w[2]=1
#print("Weights: ",w)

print("\nEnsembling the three models with respective weights...")
def ensemble(x,weights,models):     
    outputs=[]
    for model in models:                
        outputs.append(list(model.predict(x)[0]))                
    
    outputs=np.array(outputs)
    avg=np.average(a=outputs,axis=0,weights=weights)
    return avg


predictions=[]
print("Predicting test images with ensemble model...")
pbar=ProgressBar()
for i in pbar(range(len(test_images))):
  pred=ensemble(test_images[i].reshape(-1,img_size,img_size,3),w,models)
  predictions.append(pred)



def equal(pred,label):
  if pred[0]>=0.5:
      pred_class=0
  else:
      pred_class=1
  if pred_class==label:
    return True
  else:
    return False

def accuracy(predicted_values,labels):
  
  total=len(labels)
  correct=0
  for i in range(len(labels)):
    if equal(predicted_values[i],labels[i]):
      correct+=1
  print("Correctly predicted",correct,"images out of",total)
  acc=correct/total
  return acc

print("Accuracy: ",accuracy(predictions,test_labels))

y_pred=np.argmax(np.array(predictions),axis=1)

print("Classification Report:")
print(classification_report(y_pred=y_pred,y_true=test_labels))
print("Confusion Matrix:")
print(confusion_matrix(y_pred=y_pred,y_true=test_labels))
