import sys
from utils import *
import numpy as np


def generate_reverse_lookup(vocab):
    reverse_vocab = {}
    vocab_list = vocab.keys()
    for key in vocab_list:
        index = vocab[key][0]
        reverse_vocab[index] = key
    return reverse_vocab


if __name__ == '__main__':

    config = Config('config.ini')
    vocab, _, _ = get_vocabulary(config)

    reverse_vocab = generate_reverse_lookup(vocab)

    embedding_file_name = sys.argv[1]
    embeddings = np.loadtxt(embedding_file_name, delimiter=",")

    word = sys.argv[2]
    word_id = vocab[word][0]
    word_embedding = embeddings[word_id]

    similarity_array = []
    for embedding in embeddings:
        similarity_array.append(np.dot(word_embedding, embedding))

    result = [i[0] for i in sorted(enumerate(similarity_array), key=lambda x:-1*x[1])]

    print("Words similar to " + str(word))
    print("\n".join(map(lambda x: reverse_vocab[x], result[1:31])))
