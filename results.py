import csv
from driver import *
import numpy as np
from scipy.stats.stats import pearsonr


def get_embed_score(embed, index1, index2):
    return np.dot(embed[index1], embed[index2])


def get_scores():

    vocab, _, _ = load_vocabulary('default.stop.vocab')
    embed = np.loadtxt("emb.1.w_5.stop_words.neg_120.out", delimiter=",")

    f = open('results.txt', 'w')

    simlex_vector = []
    embed_vector = []

    with open("SimLex-999.txt") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            w1 = line[0]
            w2 = line[1]
            simlex_score = line[3]

            if w1 not in vocab or w2 not in vocab:
                continue

            embed_score = get_embed_score(embed, vocab[w1][0], vocab[w2][0])

            f.write(w1 + "," + w2 + "," + str(simlex_score) + "," + str(embed_score) + "\n")

            simlex_vector.append(float(simlex_score))
            embed_vector.append(embed_score)

    print(pearsonr(simlex_vector, embed_vector))


if __name__ == '__main__':
    get_scores()
