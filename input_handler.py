from utils import *
import numpy as np
from random import randint


class DataHandler:

    document_ids = None
    words = None

    document_count = 0
    sentence_count = 0
    word_count = 0

    document_index = 0
    sentence_index = 0
    word_index = 0

    negative_samples = 0
    window_length = 0

    vocabulary = None
    vocabulary_size = 0
    word_occurrences_count = 0
    reverse_vocab = None

    def __init__(self, vocabulary, vocabulary_size, word_occurrences_count, negative_samples, window_length):

        self.vocabulary = vocabulary
        self.vocabulary_size = vocabulary_size
        self.word_occurrences_count = word_occurrences_count
        self.negative_samples = negative_samples
        self.window_length = window_length
        self.generate_reverse_vocab()

        self.document_ids = get_document_names()
        self.document_count = len(self.document_ids)
        self.load_document()

        print("Total files " + str(self.document_count))

    def reset(self):
        self.document_index = 0
        self.sentence_index = 0
        self.word_index = 0
        self.load_document()

    def generate_reverse_vocab(self):
        self.reverse_vocab = {}
        vocab_list = self.vocabulary.keys()
        for key in vocab_list:
            index = self.vocabulary[key][0]
            self.reverse_vocab[index] = key

    def load_document(self):
        if self.document_index % 100 == 0:
            print("Current file index - " + str(self.document_index))

        if self.document_index == self.document_count:
            return

        sentences = get_sentences(self.document_ids[self.document_index])
        self.words = []
        for sentence in sentences:
            words = get_words(sentence)
            if len(words) != 0:
                self.words.append(get_words(sentence))

        self.sentence_count = len(self.words)
        self.word_count = len(self.words[0])

        self.sentence_index = 0
        self.word_index = 0

    def get_probabilities(self, encodings):
        result = []
        for val in encodings:
            word = self.reverse_vocab[val]
            count = self.vocabulary[word][1]
            result.append(count/self.word_occurrences_count)
        return result

    def get_negative_samples(self, input_enc, neighbour_enc):
            result = set()
            while len(result) < self.negative_samples:
                random_emb = randint(0, self.vocabulary_size - 1)
                if random_emb == input_enc or random_emb == neighbour_enc or random_emb in result:
                    continue
                result.add(random_emb)
            return list(result)

    def get_next(self):
        # check for end of sentence
        if self.word_index == self.word_count:
            self.sentence_index = self.sentence_index + 1
            self.word_index = 0

            if self.sentence_index == self.sentence_count:
                self.document_index += 1
                self.load_document()
            else:
                self.word_count = len(self.words[self.sentence_index])

            if self.document_index == self.document_count:
                return None

        w_index = self.word_index
        s_index = self.sentence_index

        temp = self.words[s_index]
        # print(str(w_index) + " " + str(s_index) + " " + str(self.word_count) + " " + str(self.sentence_count))
        # print(" ".join(self.words[s_index]))
        word = temp[w_index]
        neighbours = []
        for i in range(max(0, w_index - self.window_length), min(self.word_count, w_index + self.window_length)):
            if i == w_index:
                continue
            neighbours.append(self.words[s_index][i])

        data = []
        input_enc = self.vocabulary[word][0]
        for neighbour in neighbours:
            neighbour_enc = self.vocabulary[neighbour][0]
            negative_sample_enc = self.get_negative_samples(input_enc, neighbour_enc)
            probabilities = self.get_probabilities(negative_sample_enc)
            data.append({
                'word': np.asarray([input_enc]),
                'label': np.asarray([neighbour_enc]),
                'neg': np.asarray(negative_sample_enc),
                'prob': np.asarray(probabilities).reshape(self.negative_samples, 1)
            })
        self.word_index = self.word_index + 1
        return data
