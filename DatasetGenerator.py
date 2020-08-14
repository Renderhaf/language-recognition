import WordSuppplier, json
import numpy as np

language_codes = {
    'english': 'en',
    'spanish': 'es'
}

def getPages(language: str) -> list:
    with open('articles.json', 'r') as file:
        articles: dict = json.loads(file.read())
    return articles.get(language_codes.get(language, language), [])

def fixWords(words: list):
    special_letters = {
        'á' : 'a', 'é' : 'e', 'í' : 'i', 'ó': 'o', 'ú' : 'u', 'ü' : 'u', 'ñ': 'n'
    }

    for i in range(len(words)):
        nword = ''
        for j in range(len(words[i])):
            words[i] = words[i].lower()

            if words[i][j] in special_letters.keys():
                nword += special_letters.get(words[i][j])
            else:
                nword += words[i][j]
        words[i] = nword

def getAllLanguageWords(language: str):
    page_names = getPages(language)
    words = set()
    for pn in page_names:
        page_words = WordSuppplier.getWordsFromArticle(pn, language_codes.get(language, language))
        fixWords(page_words)
        for word in page_words:
            words.add(word)
    return list(words)

def generateAllWords():
    with open('articles.json', 'r') as file:
        articles: dict = json.loads(file.read())
    
    all_words = {
        k : getAllLanguageWords(k) for (k,v) in articles.items()
    }
    
    return all_words    

def makeWordVector(word:str, max_length: int):
    '''
    Convert every word into a list with length @param max_length
    every element in the list represents a char in the @param word
    as a list, where every element is 0 except for the correct char 
    (where the correct char is at its location in the alphabet / ord(char) - ord('a'))

    In addition, an empty char is represented as a list where every element is 0
    '''
    word_vector = []
    for i in range(max_length):
        if i >= len(word):
            word_vector.append([0 for j in range(26)]) # Add an empty char if the string is not the longest
        else:
            word_vector.append([1 if j == ord(word[i])-ord('a') else 0 for j in range(26)]) # Add the char as a vector if you can
    return word_vector
    
def convertDictToDataset(all_words: dict)->list:
    max_length = -1
    max_word = ''
    for words in all_words.values():
        for word in words:
            if max_length < len(word):
                max_length = len(word)
                max_word = word

    values = []
    results = []
    for lang_index, language in enumerate(all_words.keys()): #Over every language
        for word in all_words[language]: #over every word
            word_vector = makeWordVector(word, max_length)
            values.append(word_vector)
            results.append([1 if i == lang_index else 0 for i in range(len(all_words.keys()))]) # Represent the language with onehot-enconding

    return values, results

def makeTrainingDataFile():
    values, results = convertDictToDataset(generateAllWords())
    data = [values, results]
    np.save("training_data.npy", data)

if __name__ == "__main__":
    makeTrainingDataFile()
