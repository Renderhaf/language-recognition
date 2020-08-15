from NeuralNet import NeuralNet
import numpy as np
from DatasetGenerator import makeFlatWordVector, language_codes

model = NeuralNet(17, 2)


def predict_words():
    raw_word = input("Enter a word (enter an empty string to stop): ")

    while raw_word != "":

        word = np.array(makeFlatWordVector(raw_word, 17))

        prediction = np.array(model.predict(word))
        pred_language = list(language_codes.keys())[prediction.argmax()]
        pred_precentage = round(float(prediction[prediction.argmax()]), 3)*100

        print(
            f"The model predicted {pred_language} with {pred_precentage}% certainty")

        raw_word = input("Enter a word (enter an empty string to stop): ")


def predict_sentences():
    raw_sentence = input("Enter a sentence (enter an empty string to stop): ")

    while raw_sentence != "":

        predictions = []
        for raw_word in raw_sentence.split():
            word = np.array(makeFlatWordVector(raw_word, 17))
            prediction = np.array(model.predict(word))
            predictions.append(prediction)

        # Get the avarage prediction for each language in the sentence
        final_predictions = np.array([sum(lang_predictions)/len(predictions)
                             for lang_predictions in zip(*predictions)])

        pred_language = list(language_codes.keys())[final_predictions.argmax()]
        pred_precentage = round(float(final_predictions[final_predictions.argmax()]), 3)*100

        print(
            f"The model predicted {pred_language} with {pred_precentage}% certainty")

        raw_sentence = input(
            "Enter a sentence (enter an empty string to stop): ")

if __name__ == "__main__":
    choice = int(input('Do you want to predict a word (0), a sentence (1) or exit(2)? '))
    if choice == 0:
        predict_words()
    elif choice == 1:
        predict_sentences()
