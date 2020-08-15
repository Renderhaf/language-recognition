import keras
import numpy as np
import os


class NeuralNet():
    def __init__(self, max_word_length, language_amount, weights_filename='weight.h5'):
        self.weights_filename = weights_filename

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(200,
                                          input_dim=max_word_length*26, activation='sigmoid'))

        # 3 layers with the sigmoid activation function
        self.model.add(keras.layers.Dense(150, activation='sigmoid'))
        self.model.add(keras.layers.Dense(125, activation='sigmoid'))
        self.model.add(keras.layers.Dense(100, activation='sigmoid'))

        # Use softmax as the output so that the languages will be a precetage
        self.model.add(keras.layers.Dense(
            language_amount, activation='softmax'))

        if os.path.isfile(f'./{self.weights_filename}'):
            print("Loading weights from file...")
            self.model.load_weights(self.weights_filename)

        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(f"Build model with {max_word_length} as the max word length and {language_amount} languages. Summery is:")
        print(self.model.summary())

    def fit(self, train_values, train_results, validation_values, validation_results, ephocs=100):
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.weights_filename, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        ]

        self.model.fit(train_values, train_results, batch_size=500, epochs=ephocs, validation_data=(
            validation_values, validation_results), callbacks=callbacks)
