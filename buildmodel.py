import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TRAIN_DIR = "/home/rdarshan927/Documents/Machine Learning/Original/Train"
TEST_DIR = "/home/rdarshan927/Documents/Machine Learning/Original/Test"
VAL_DIR = "/home/rdarshan927/Documents/Machine Learning/Original/Validate"

# Data augmentation and preparation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_set = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_set = val_datagen.flow_from_directory(
    VAL_DIR, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=[224,224,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

# Flatten before the Dense layer
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=512, activation='relu'))

# The last layer
model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

print(model.summary())


#compile the model

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )

history = model.fit(x=train_set, validation_data=val_set, batch_size=32, epochs=20)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


print(acc)
print(val_acc)


# display results

epochs_range = range(20)   # sequence of number from 0 to 20

plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = "Training Accuracy")
plt.plot(epochs_range, val_acc, label = "Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = "Training Loss")
plt.plot(epochs_range, val_loss, label = "Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and validation Loss')

plt.show()


# save the model

model.save('/home/rdarshan927/Documents/Machine Learning/Original/flowers.h5')