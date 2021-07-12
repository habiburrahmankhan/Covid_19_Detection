import tensorflow as tf 
from tensorflow.keras.applications import ResNet50V2,DenseNet201,InceptionV3
from tensorflow.keras.optimizers import Adam
import pandas as pd

def train_model(path,train_images,train_labels, 
                valid_images,valid_labels,model_name,
                epochs,input_shape,classes,callbacks):

    if model_name=='resnet50_v2':
        current_model=ResNet50V2(weights=None,include_top=False,input_shape=input_shape)
                 
    if model_name=='densenet201':
        current_model=DenseNet201(weights=None,include_top=False,input_shape=input_shape)
          
    if model_name=='inception_v3':
        current_model=InceptionV3(weights=None,include_top=False,input_shape=input_shape)


    x=current_model.output         
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    output=tf.keras.layers.Dense(classes,activation='softmax')(x)


    model=tf.keras.Model(inputs=current_model.input,outputs=output)

    optimizer=Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    
    print("Training",model_name,":")
    
    results=model.fit(train_images,train_labels,epochs=epochs,
                      validation_data=(valid_images,valid_labels), 
                      batch_size=16, 
                      callbacks=callbacks)

    losses=pd.DataFrame(model.history.history)
    losses[['loss','val_loss']].plot()
        
    save_model=path+model_name+'.h5'
    model.save(save_model)
    
    return results    