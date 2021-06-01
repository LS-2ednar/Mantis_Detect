# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:50:24 2021
"""



"""
Import used modules
"""
import os                                              #for path controls
import numpy as np                                     #to work with images
import matplotlib.pyplot as plt                        #to plot images
import tensorflow as tf                                #to generate the CNN
from PIL import Image                                  #to read images
import pathlib                                         #to acces paths 
from tensorflow.keras.callbacks import EarlyStopping   #allows EarlyStopping
import seaborn as sns                                  #allows confusionmatirx
"""
Utility Functions
"""

def load_and_prep_data(folder,newSize,seed = True,split=0.2):
    """
    load all images from a folder and rescale to get training and testing data
    """  
    #seed generation
    if seed == True:
        seed1 = np.random.randint(0,1337)
        seed2 = np.random.randint(0,1337)
        print(f'Training Seed: {seed1}\nValidation Seed: {seed2}\n')
    else:
        seed1 = 123
        seed2 = 124
    
    data_dir = pathlib.Path(folder)
    
    training = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=(split),
        subset=('training'),
        seed = seed1,
        image_size=(newSize,newSize))
    
    validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=(split),
        subset=('validation'),
        seed = seed2,
        image_size=(newSize,newSize))
    
    return training, validation

def visualization_dataset(dataset,isTraining=False): 
    """
    This function takes a Tensorflow Kears dataset and plots 16 images of the 
    Dataset
    """
    plt.figure(figsize=(20, 20))    
    for images, labels in dataset.take(1):
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[labels[i]],fontname='Arial',fontsize=32)
            plt.axis("off")
    if isTraining == True:
        plt.suptitle('Example of Training Data',fontsize=50,fontname='Arial')
    else:
        plt.suptitle('Example of Validation Data',fontsize=50,fontname='Arial')
    plt.show()
    
    return

def prediction_visualization(img,Model,training_data,show=True):
    """
    This function takes an image (img) a trained Model and the training or 
    validation data used to train the model make a prediction based on the 
    trained CNN and returns the predicted image with the predicted class and a 
    confidence value. Further the class_index is stored.    
    """
    labels = training_data.class_names
    img = Image.open(img).resize((128, 128))
    img = np.array(img)
    Prediction = Model.predict(img[None,:,:])
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    if show == True:
        plt.title(f'Prediction: {labels[Prediction.argmax(axis=-1)[0]]}\nConfidence: {round(max(Prediction[0])*100,2)}')
        plt.show()
    else:
        plt.title(f'Prediction: {labels[Prediction.argmax(axis=-1)[0]]}\nConfidence: {round(max(Prediction[0])*100,2)}'
                  ,fontsize=30)
    
    class_index = Prediction[0].argmax(axis=-1)
    
    return fig, class_index
    
def modelanalytics(history):
    """
    Plots the Modelhistory in apresentable way. 
    The cols handel is used to visualy differentiate between the two models
    """
    #color selection for the plots
    color = ['#006600','#090C85','#A6D08A','#6699FF']
    
    fig, ax1 = plt.subplots(figsize=(7.5,5))
    
    #first axis for Loss
    ax1.set_xlabel('Epoch [-]', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    Ins1 = ax1.plot(history.history['loss'], '--', color = color[0],
                    label='loss')
    Ins2 = ax1.plot(history.history['val_loss'], ':', color = color[1] , 
                    label = 'val_loss')
    ax1.tick_params(axis='y')
    ax1.set_xlim([0, len(history.history['loss'])])
    
    #second axis for Accuracy
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Accuracy', fontsize=14)
    Ins3 = ax2.plot(history.history['accuracy'], '--', color = color[2], 
                    label='accuracy')
    Ins4 = ax2.plot(history.history['val_accuracy'], ':', color = color[3], 
                    label = 'val_accuracy')
    ax2.tick_params(axis='y')
    
    #get one legend
    Ins = Ins1+Ins2+Ins3+Ins4
    labs = [l.get_label() for l in Ins]
    ax2.legend(Ins, labs, loc='best', bbox_to_anchor=(1.05, 1), fontsize=14)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    #set a title
    ax1.set_title(f'Analytics: {history.model.name}',fontsize=18)
    
    #note best val_acc and acc where the val_loss was minimal
    values = history.history['val_loss']
    index = values.index(min(values))
    print(f'At Epoch: {index}\nAccuracy: {round(history.history["accuracy"][index]*100,2)}%\nVal_Accuracy: {round(history.history["val_accuracy"][index]*100,2)}%')
    plt.gcf().text(0.72, 0.2, f'At Epoch:       {index}\nAccuracy:       {round(history.history["accuracy"][index]*100,2)}%\nVal_Accuracy: {round(history.history["val_accuracy"][index]*100,2)}%', fontsize=12)
    plt.show()
    
    return fig

def modelperformance(test_images,test_labels,Model,training_data,only_confusion = False):
    """
    Function to test model classification using a confusion matrix
    """
    if only_confusion == False:
        figure = plt.figure(figsize=(28, 28))
        figure.suptitle(f'{Model.name} Performance', fontsize= 72)
    else:
        figure = plt.figure(figsize=(8,8))
    #initialization of prediction_labels
    pred_labels = []
    
    if only_confusion == False:
        i = 0
        for image in test_images:
            plt.subplot(3,3,i+1)
            fig, classi = prediction_visualization(image,Model,training_data,show=False)
            pred_labels.append(classi)
            i += 1
    else:
        for image in test_images:
            fig, classi = prediction_visualization(image,Model,training_data,show=True)
            pred_labels.append(classi)
        
    cf_matrix = tf.math.confusion_matrix(test_labels, pred_labels)
    if only_confusion == False:
        plt.subplot(3,3,i+1)
    
    categories = training_data.class_names
    
    if len(training_data.class_names)<5:
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                                fmt='.2%', 
                                xticklabels= categories, 
                                yticklabels= categories,
                                linewidths=.5,
                                cmap='Blues')
    else:
        sns.heatmap(cf_matrix, annot=False,
                    xticklabels= categories,
                    yticklabels= categories,
                    linewidths=.5, 
                    cmap='Blues')
    
    plt.show()
    return figure
    

"""
Setup the Working Enviornment
"""
#path to file
abspath = os.path.abspath(__file__)
#path to folder the file is in
dname = os.path.dirname(abspath)
#change working directory to file location
os.chdir(dname)

"""
*--------------------------------------*
*--------------------------------------*
1: CNN Model to Detect a Mantis or Plant
*--------------------------------------*
*--------------------------------------*
"""
"""
Loading Data
"""

img_dim = 128
training, validation = load_and_prep_data('./Augmented_Data_2C_200',img_dim,split=0.2)
visualization_dataset(training, True)
visualization_dataset(validation)

"""
CNN Model 2 Classes
"""
Model = tf.keras.models.Sequential(name='MantisDetect_2C')
Model.add(tf.keras.layers.Conv2D(200,(1,1),padding='same',activation='sigmoid', 
                                 input_shape=(img_dim,img_dim,3))) 
Model.add(tf.keras.layers.AveragePooling2D(pool_size= (4,4)))
Model.add(tf.keras.layers.Dropout(0.01))
Model.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model.add(tf.keras.layers.AveragePooling2D(pool_size= (4,4)))
Model.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2)))
Model.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model.add(tf.keras.layers.Dropout(0.01))
Model.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2)))
Model.add(tf.keras.layers.Conv2D(20,(1,1),padding='same',activation='swish'))
Model.add(tf.keras.layers.Flatten())
Model.add(tf.keras.layers.Dropout(0.01))
Model.add(tf.keras.layers.Dense(2,activation='softmax'))  #Classes

#optimizer and learning rate determination
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

Model.summary()

"""
EarlyStopping monitor to prevent the model from overfittig
"""
monitor = EarlyStopping(monitor='val_loss', 
                        min_delta=1e-3, 
                        patience = 20, 
                        verbose=1, 
                        restore_best_weights=True)

"""
Training & Validation
"""
history = Model.fit(training, 
                    validation_data = validation, 
                    callbacks=[monitor],
                    epochs=1000)
"""
Save or Load Model
"""
# Model.save('Model_2C')
# Model = tf.keras.models.load_model('Model_2C')

"""
Analysis of Model without Data Augmentation
"""
modelanalytics(history)

"""
Predicting an Image with the new Model
"""
folder = 'Test_Images'
tests = os.listdir(folder)
for test in tests:
    prediction_visualization(os.path.join(folder,test), Model, training)


"""
*-------------------------------------------------------*
*-------------------------------------------------------*
2: CNN Model to Detect a different Mantistypes and Plants 
*-------------------------------------------------------*
*-------------------------------------------------------*
"""
"""
Loading Data
"""

img_dim = 128
training2, validation2 = load_and_prep_data('./Augmented_Data_17C_50',img_dim,split=0.2)
visualization_dataset(training2, True)
visualization_dataset(validation2)

"""
CNN Model 17 Classes
"""
Model2 = tf.keras.models.Sequential(name='MantisDetect_17C')

Model2.add(tf.keras.layers.Conv2D(200,(1,1),padding='same',activation='sigmoid', 
                                  input_shape=(img_dim,img_dim,3))) 
Model2.add(tf.keras.layers.AveragePooling2D(pool_size= (4,4)))
Model2.add(tf.keras.layers.Dropout(0.01))
Model2.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model2.add(tf.keras.layers.AveragePooling2D(pool_size= (4,4)))
Model2.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model2.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2)))
Model2.add(tf.keras.layers.Conv2D(20,(2,2),padding='same',activation='swish'))
Model2.add(tf.keras.layers.Dropout(0.01))
Model2.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2)))
Model2.add(tf.keras.layers.Conv2D(20,(1,1),padding='same',activation='swish'))
Model2.add(tf.keras.layers.Flatten())
Model2.add(tf.keras.layers.Dropout(0.01))
Model2.add(tf.keras.layers.Dense(17,activation='softmax'))  #Classes

#optimizer and learning rate determination
Model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

Model2.summary()

"""
Training & Validation to avoid overfitting Earlystoping is implemented aswell
"""
history2 = Model2.fit(training2, 
                    validation_data = validation2, 
                    callbacks=[monitor],
                    epochs=1000)
"""
Save or Load Model 
"""
# Model2.save('Model_17C')
# Model2 = tf.keras.models.load_model('Model_17C')

"""
Analysis of Model without Data Augmentation
"""
modelanalytics(history2)

"""
Predicting an Image 
"""
folder = 'Test_Images'
tests = os.listdir(folder)
for test in tests:
    prediction_visualization(os.path.join(folder,test), Model2, training2)

"""
Model Performance Comparison  
"""
#Folder for Performance Test images
folder = 'Performance_Images'
parts = os.listdir(folder)
Tests = []
for part in parts:
    Tests.append(os.path.join(folder,part))

#known labels for the tested images
labelsM1 = [0,0,0,0,1,1,1,1]  #  2-classes case
labelsM2 = [6,3,5,4,7,16,0,2] # 17-classes case

modelperformance(Tests,labelsM1,Model,training)   #  2-classes model 
modelperformance(Tests,labelsM2,Model2,training2) # 17-classes model
