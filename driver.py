import argparse
from utils import *
from model import *
import numpy as np


def create_vocabulary():
    tokens = {}
    size = 0
    count = 0
    document_ids = get_document_names()
    for document_id in document_ids:
        sentences = get_sentences(document_id)
        for sentence in sentences:
            words = get_words(sentence)
            for word in words:
                count = count + 1
                if word not in tokens:
                    tokens[word] = 1
                    size = size + 1
                else:
                    tokens[word] = tokens[word] + 1

    vocabulary = {}
    for index, key in enumerate(tokens.keys()):
        vocabulary[key] = (index, tokens[key])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--vocabulary-file', dest='vocabulary_file', type=str,
                        help='input vocabulary file name', default=None)
    args = parser.parse_args()

    vocabulary = {}
    v_size = 0
    v_count = 0
    if args.vocabulary_file is None:
        vocabulary, v_size, v_count = create_vocabulary()
    else:
        try:
            vocabulary, v_size, v_count = load_vocabulary(args.vocabulary_file)
        except FileNotFoundError:
            vocabulary, v_size, v_count = create_vocabulary()
            save_vocabulary(vocabulary, args.vocabulary_file)

    print("Loaded Vocabulary")

    dim = 60
    window = 5
    learning_rate = 0.5
    negative_samples = 120

    print("Defining Model")
    input_enc, label, neg_samples, prob, embeddings, loss = get_model(negative_samples, dim, v_size)
    print("Defined Model")
    print("Initializing DataHandler")
    data_handler = DataHandler(vocabulary, v_size, v_count, negative_samples, window)
    print("Initialized DataHandler")
    print("Starting training")
    final_embeddings = run(input_enc, label, neg_samples, prob, embeddings, loss, data_handler)

    np.savetxt('emb.1.w_5.stop_words.neg_120.out', final_embeddings, delimiter=',')
