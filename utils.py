import nltk
import re
import random
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from config_handler import Config
stopWords = set(stopwords.words('english'))


# False for lt as it's encoding issue.
# False for stop words
# False words less than size 2
def is_valid_token(word):
    if word in stopWords:
        return False
    if word == "lt":
        return False
    if len(word) < 2:
        return False
    return True


# get training validation and test split
def get_document_names(config: Config):
    try:
        with open(config.files_split_sourcefile, 'r') as f:
            training, validation, test = f.read().splitlines()
            return training.split(","), validation.split(","), test.split(",")
    except FileNotFoundError:
        raise Exception(config.files_split_sourcefile + "(data split information file) doesn't exist.\n"
                                                      + "Lookup readme for more info\n")


# gets vocab from source mentioned in config
def get_vocabulary(config: Config):
    try:
        return load_vocabulary(config.vocabulary_sourcefile)
    except FileNotFoundError:
        raise Exception(config.files_split_sourcefile + "(source vocabulary file) doesn't exist.\n"
                        + "Lookup readme for more info\n")


# split corpus to generate training and validation documents
def split_corpus():
    training_files = get_training_document_names()
    test_ids = get_test_document_names()

    random.shuffle(training_files)

    training_ids = training_files[: int(0.9*len(training_files))]
    validation_ids = training_files[int(0.9*len(training_files)):]

    return training_ids, validation_ids, test_ids


def get_test_document_names():
    return list(filter(lambda x: re.match(r'^test', x), reuters.fileids()))


def get_training_document_names():
    return list(filter(lambda x: re.match(r'^training', x), reuters.fileids()))


def get_sentences(document_id):
    raw = reuters.raw(document_id)
    return sent_tokenize(raw.replace("\n", ".\n", 1))


# replace non alphabets with spaces
# tokenize -> strip -> lowercase and check if it's valid word.
def get_words(sentence):
    sentence = re.sub(r'[^a-zA-Z.]', ' ', sentence)
    words = nltk.word_tokenize(sentence)
    words = map(lambda w: w.strip().strip('.').lower(), words)
    return list(filter(lambda w: is_valid_token(w), words))


def create_vocabulary(document_ids):
    tokens = {}
    size = 0
    count = 0
    for document_id in document_ids:
        sentences = get_sentences(document_id)
        for sentence in sentences:
            words = get_words(sentence)
            for word in words:
                count = count + 1
                if word not in tokens:
                    tokens[word] = 1
                else:
                    tokens[word] = tokens[word] + 1

    vocabulary = {}
    for key in tokens.keys():
        if tokens[key] > 1:
            vocabulary[key] = (size, tokens[key])
            size += 1
    return vocabulary, size, count


def save_vocabulary(vocabulary, filename):
    f = open(filename, 'w')
    for k in vocabulary.keys():
        f.write(k + "," + str(vocabulary[k][0]) + "," + str(vocabulary[k][1]) + "\n")


def load_vocabulary(filename):
    vocabulary = {}
    size = 0
    count = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(",")
            if len(elements) != 3:
                continue
            vocabulary[elements[0]] = (int(elements[1]), int(elements[2]))
            size = size + 1
            count = count + int(elements[2])
    return vocabulary, size, count


def compile_results(config, training_stats, avg_test_loss, files):
    return str(config)\
           + "Training stats per epoch\n"\
           + "\n".join(map(lambda x: "Avg. Training Loss - " + str(x[0]) + " Avg. Validation Loss - " + str(x[1]), training_stats)) + "\n\n"\
           + "Avg Test Loss - " + str(avg_test_loss) + "\n\n"\
           + "Embeddings Generated\n"\
           + "\n".join(files)
