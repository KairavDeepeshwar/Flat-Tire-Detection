import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2 as cv
import os, sys
import pymp

path1="C:\\Users\\meera\\Documents\\BTech CSE spl. AIML\\VS code\\CAO Project\\cao_dataset\\flat.class"
path2="C:\\Users\\meera\\Documents\\BTech CSE spl. AIML\\VS code\\CAO Project\\cao_dataset\\full.class"
path3="C:\\Users\\meera\\Documents\\BTech CSE spl. AIML\\VS code\\CAO Project\\cao_dataset\\no-tire.class"
files1=os.listdir(path1)
files2=os.listdir(path2)
files3=os.listdir(path3)
Num_files_FLT=len(files1)
Num_files_FUT=len(files2)
Num_files_NT=len(files3)
dataset_len=Num_files_FLT+Num_files_FUT+Num_files_NT

len(files3)

data=np.zeros((dataset_len,100,100,1))
label=[]
data.shape

#------------Parallel Processing-------------------------
def load_and_preprocess_image(image_path, data, label, index, class_label):
    img = cv.imread(image_path)
    img_gs = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_gs = cv.resize(img_gs, (100, 100))
    img_gs = img_gs / 255
    img_gs = img_gs.reshape(100, 100, 1)
    
    with data.lock:
        data[index, :, :] = img_gs
        label.append(class_label)

with pymp.Parallel(4) as p:
    for i in p.range(Num_files_FLT):
        image_path = os.path.join(path1, files3[i])
        load_and_preprocess_image(image_path, data, label, i, 'Flat Tire')

with pymp.Parallel(4) as p:
    for i in p.range(Num_files_FUT):
        image_path = os.path.join(path2, files2[i])
        load_and_preprocess_image(image_path, data, label, i + Num_files_FLT, 'Full Tire')

with pymp.Parallel(4) as p:
    for i in p.range(Num_files_NT):
        image_path = os.path.join(path3, files3[i])
        load_and_preprocess_image(image_path, data, label, i + Num_files_FLT + Num_files_FUT, 'No Tire')
#------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
lab=le.fit_transform(label)

label
lab
train_images,test_images,train_labels,test_labels=train_test_split(data,lab,test_size=0.2,random_state=1)

print('Train Dataset Size:',np.size(train_labels))
print('Test Dataset Size:',np.size(test_labels))

network=models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(100,100,1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(7,7),activation='relu'))
network.add(layers.Conv2D(32,(3,3),activation='relu'))
network.add(layers.MaxPooling2D((3,3)))

network.summary()

network.add(layers.Flatten())
network.add(layers.Dense(60,activation='relu'))
network.add(layers.Dense(90,activation='relu'))
network.add(layers.Dense(3,activation='softmax'))

network.summary()

network.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

test_labels

trained_network=network.fit(train_images,train_labels,epochs=50,validation_data=(test_images,test_labels))

train_labels.shape

plt.plot(trained_network.history['accuracy'],label='Training Accuracy')
plt.plot(trained_network.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1.1])
plt.legend(loc='lower right')

plt.plot(trained_network.history['loss'],label='Training Loss')
plt.plot(trained_network.history['val_loss'],label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

test_loss,test_acc=network.evaluate(test_images,test_labels)

y_predict=network.predict(test_images)

y_pred=[]
for val in y_predict:
    y_pred.append(np.argmax(val))

print("Accuracy:",(metrics.accuracy_score(test_labels,y_pred))*100,"%")

user_input_path = input("Enter the path of the image you want to classify: ")

user_image = cv.imread(user_input_path)
user_image_gs = cv.cvtColor(user_image, cv.COLOR_RGB2GRAY)
user_image_gs = cv.resize(user_image_gs, (100, 100))
user_image_gs = user_image_gs / 255
user_image_gs = user_image_gs.reshape(1, 100, 100, 1)

predicted_class = network.predict(user_image_gs)
predicted_label = le.inverse_transform([np.argmax(predicted_class)])[0]

print(f"The predicted class is: {predicted_label}")