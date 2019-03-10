import csv
from driver import *
import numpy as np
from scipy.stats.stats import pearsonr
import sys


def get_embed_score(embed, index1, index2):
    return np.dot(embed[index1], embed[index2])


def get_scores(vocab, embeddings, category, threshold):

    simlex_vector = []
    embed_vector = []

    with open("data/SimLex-999.txt") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            w1 = line[0]
            w2 = line[1]
            pos = line[2]
            simlex_score = line[3]

            if category is not None and pos != category:
                continue
            if w1 not in vocab or w2 not in vocab:
                continue
            if threshold is not None and min(vocab[w1][1], vocab[w2][1]) < threshold:
                continue

            embed_score = get_embed_score(embeddings, vocab[w1][0], vocab[w2][0])

            # print(w1 + " " + w2 + " " + str(simlex_score) + " " + str(embed_score))

            simlex_vector.append(float(simlex_score))
            embed_vector.append(embed_score)

    print(len(embed_vector))
    score = pearsonr(simlex_vector, embed_vector)[0]
    print("Count - " + str(len(embed_vector)) + (" Type - " + category if category is not None else "") + " Score - " + str(score))


if __name__ == '__main__':

    config = Config('config.ini')
    vocab, _, _ = get_vocabulary(config)

    threshold = None
    if len(sys.argv) > 2:
        threshold = int(sys.argv[2])

    embedding_file_name = sys.argv[1]
    embeddings = np.loadtxt(embedding_file_name, delimiter=",")

    get_scores(vocab, embeddings, "N", threshold)
    get_scores(vocab, embeddings, "V", threshold)
    get_scores(vocab, embeddings, "A", threshold)
    get_scores(vocab, embeddings, None, threshold)

