import math
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds;
import matplotlib


def norlmalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# for images
dataset, metadata = tfds.load("fashion_mnist", as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.map(norlmalize)

#model for image
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(60000).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(6000/BATCH_SIZE))

print(model.evaluate(test_dataset,steps=math.ceil(60000/32)))

class_names = metadata.features['label'].names
print(class_names)

