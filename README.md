# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

### STEP 2:

### STEP 3:

Write your own steps

## PROGRAM
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train.shape

X_test.shape

single_image= X_train[800]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

y_train[800]

X_train.max()

X_train.min()



X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[800]
plt.imshow(single_image,cmap='gray')

y_train_onehot[800]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

X_train_scaled.shape

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics='accuracy')


model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

**Prediction for a single input**



img = image.load_img('ppp.png')

type(img)

img = image.load_img('ppp.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

   print(x_single_prediction)



```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
### ACCURACY VS VAL_ACCURACY
![download](https://user-images.githubusercontent.com/75235150/191758267-0e5f4600-6f90-4133-9c9e-9fddc6d7b320.png)



### TRAINING_LOSS VS VAL_LOSS
![download (1)](https://user-images.githubusercontent.com/75235150/191758354-231e8960-f35a-432c-8fd4-d0d7d60a89e0.png)


### Classification Report

![image](https://user-images.githubusercontent.com/75235488/189907255-1bf07e4b-645d-4643-b9f8-a910dc2ea19b.png)

### Confusion Matrix
![image](https://user-images.githubusercontent.com/75235488/189907192-8b5c23c9-27c1-40d3-8499-7bd8091f3c76.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235488/189907327-69384ddd-16ed-4ba2-95c1-6e99e6dbaf63.png)

### Predicted Output

![Screenshot 2022-11-10 172901](https://user-images.githubusercontent.com/75235150/201085699-62c8b50c-cca8-4116-a079-2018563ba1a0.png)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
