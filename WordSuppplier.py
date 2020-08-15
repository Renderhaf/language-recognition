import wikipediaapi 
import re

def _getWikiArticle(name: str, lang='en'):
    '''
    Returns the text from a wikipedia page
    parameters: page title, wikipedia language
    '''
    wiki = wikipediaapi.Wikipedia(lang)
    page = wiki.page(name).text
    return page

def _getAllWords(contents: str):
    '''
    Return all words from a string
    '''
    deleimiters = ' .,-?!\n:()[]{}"\'«»'
    words = []
    lastWordEnd = 0
    for i in range(len(contents)):
        if deleimiters.find(contents[i]) != -1:
            w = contents[lastWordEnd:i]
            if len(w) > 0:
                words.append(w)
            lastWordEnd = i+1
    return words

def _getFilteredWords(words: list):
    '''
    Filter words 
    '''
    wordlist = []
    pattern = r'^[A-Za-z]{2,}$'
    for word in words:
        if len(re.findall(pattern, word)) == 1 and word not in wordlist and 18 > len(word) > 1:
            wordlist.append(word)

    return wordlist

def getWordsFromArticle(title: str, lang='es'):
    words = _getFilteredWords(
            _getAllWords(
                _getWikiArticle(title, lang)
                )
            )

    if 'futuresgovernmentofficial' in words:
        print(title, lang)

    return words


if __name__ == "__main__":
    words = getWordsFromArticle("William Shakespeare", 'es')
    print(words[:100], len(words))

