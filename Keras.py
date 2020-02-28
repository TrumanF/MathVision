import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import GenerateData

class_names = GenerateData.characters


def make_predictions(input_img):
    model = keras.models.load_model("C:/Users/Truman/Documents/GitHub/MathVision/MODEL/Model1.h5")
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(input_img)
    predictions_list = []
    for predict in predictions:
        predictions_list.append(class_names[np.argmax(predict)])
    return predictions_list


def main():
    f = h5py.File("C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f38912_characters.h5", "r")
    f_test = h5py.File("C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f6_characters_test.h5", "r")

    dset_labels = f.get("labels").value
    dset_images = f.get("images").value

    train_images = dset_images / 255.0
    train_labels = dset_labels

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


if __name__ == '__main__':
    main()