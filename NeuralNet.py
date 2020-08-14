import keras
import numpy as np
import os


class NerualNet():
    def __init__(self, max_word_length, lenguage_amount, weights_filename='weight.h5'):
        self.weights_filename = weights_filename

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(max_word_length,
                                          input_shape=(max_word_length, 26)))

        # 4 layers with the sigmoid activation function
        self.model.add(keras.layers.Dense(200, activation='sigmoid'))
        self.model.add(keras.layers.Dense(150, activation='sigmoid'))
        self.model.add(keras.layers.Dense(125, activation='sigmoid'))
        self.model.add(keras.layers.Dense(100, activation='sigmoid'))

        # Use softmax as the output so that the languages will be a precetage
        self.model.add(keras.layers.Dense(
            lenguage_amount, activation='softmax'))

        if os.path.isfile(f'./{self.weights_filename}'):
            self.model.load_weights(self.weights_filename)

        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, train_values, train_results, validation_values, validation_results, ephocs=25):
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.weights_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        ]

        self.model.fit(train_results, train_values, batch_size=500, epochs=ephocs, validation_data=(
            validation_values, validation_results), callbacks=[callbacks])
