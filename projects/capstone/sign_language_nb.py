
#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras.models import Sequential
from keras.layers import Dense, LocallyConnected2D, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

#%% [markdown]
# # Look at Data

#%%
train = open('data/sign_mnist_train.csv')
train_df = pd.read_csv(train)
label_counts = pd.value_counts(train_df.label).to_frame()
label_counts['x'] = label_counts.index
label_counts = label_counts.rename(columns={'label': 'y'})
label_counts
sns.barplot(x='x', y='y', data=label_counts)


#%%
train = open('data/sign_mnist_train.csv')
test = open('data/sign_mnist_test.csv')

train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

# labels represents letters. 
train_y_values = train_df.label.values
test_y_values = test_df.label.values

label_binarizer = LabelBinarizer()
train_y_values = label_binarizer.fit_transform(train_y_values)
test_y_values = label_binarizer.fit_transform(test_y_values)

train_df.drop('label', axis=1, inplace=True)
test_df.drop('label', axis=1, inplace=True)


#%%
train_images = np.array([row.values.reshape(28, 28) for i, row in train_df.iterrows()])
test_images = np.array([row.values.reshape(28, 28) for i, row in test_df.iterrows()])
#train_images = train_images / 255
#test_images = test_images / 255


#%%
x_train, x_valid, y_train, y_valid = train_test_split(train_images, train_y_values, 
                                                      test_size = 0.2,
                                                      random_state = 64209)

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_valid = np.array(x_valid)
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)


#%%
train_datagen = ImageDataGenerator(shear_range = 0.25,
                                   zoom_range = 0.15,
                                   rotation_range = 15,
                                   brightness_range = [0.15, 1.15],
                                   width_shift_range = [-2,-1, 0, +1, +2],
                                   height_shift_range = [ -1, 0, +1],
                                   fill_mode = 'reflect')
test_datagen = ImageDataGenerator()

train_datagen.fit(x_train)
#train_datagen.fit(x_valid)
for image in train_datagen.flow(x_train):
    print(image)


#%%
batch_size = 64
num_classes = 24
epochs = 64


#%%
def modelMaker(dropout_rate=.20, l2_rate=.0001, dense_size=64):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(4, 4), input_shape=(28, 28, 1), padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(keras.layers.PReLU(alpha_initializer='zeros', 
                                 alpha_regularizer=None, alpha_constraint=None, 
                                 shared_axes=None))
    model.add(AveragePooling2D(pool_size = (2, 2), strides=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, kernel_size = (4, 4), padding='same', 
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(keras.layers.PReLU(alpha_initializer='zeros', 
                                 alpha_regularizer=None, alpha_constraint=None, 
                                 shared_axes=None))
    model.add(AveragePooling2D(pool_size = (2, 2), strides=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, kernel_size = (3, 3), padding='same', 
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(keras.layers.PReLU(alpha_initializer='zeros', 
                                 alpha_regularizer=None, alpha_constraint=None, 
                                 shared_axes=None))
    model.add(AveragePooling2D(pool_size = (2, 2), strides=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(dense_size, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(), metrics=['accuracy'])
    return model

model = modelMaker()


#%%
history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), 
                    epochs=epochs, batch_size=batch_size)


#%%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


#%%
test_images = np.array([i.flatten() for i in test_images])
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
y_pred = model.predict(test_images)

accuracy_score(test_y_values, y_pred.round())


#%%



