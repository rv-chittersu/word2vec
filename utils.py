import nltk
import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


def is_valid_token(word):
    if word in stopWords:
        return False
    if word == "lt":
        return False
    if len(word) < 2:
        return False
    return True


def get_document_names():
    return list(filter(lambda x: re.match(r'^training', x), reuters.fileids()))


def get_sentences(document_id):
    raw = reuters.raw(document_id)
    return raw.replace("\n", ".\n", 1).split(".\n")


def get_words(sentence):
    sentence = re.sub(r'[^a-zA-Z.]', ' ', sentence)
    words = nltk.word_tokenize(sentence)
    words = map(lambda w: w.strip().strip('.').lower(), words)
    return list(filter(lambda w: is_valid_token(w), words))
