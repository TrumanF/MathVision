import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f72886_characters.h5", "r")
f_test = h5py.File("C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f15_characters_test.h5", "r")

dset_labels = f.get("labels").value
dset_images = f.get("images").value

test_dset_labels = f_test.get("labels").value
test_dset_images = f_test.get("images").value

train_images = dset_images / 255.0
train_labels = dset_labels

test_images = test_dset_images / 255.0
test_labels = test_dset_labels

class_names = [x for x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#$()+,-.:=[]^{}']
new_model = False
try:
    model = keras.models.load_model("C:/Users/Truman/Documents/GitHub/MathVision/MODEL/Model1.h5")
    new_model = False
except OSError:
    print("Couldn't find existing model")
    new_model = True

if new_model:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(77)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    model.save("C:/Users/Truman/Documents/GitHub/MathVision/MODEL/Model1.h5")

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(77))
  plt.yticks([])
  thisplot = plt.bar(range(77), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

for i in range(16):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

