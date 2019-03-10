import sys
import numpy as np
from utils import *
from scipy import spatial


def generate_reverse_lookup(vocab):
    reverse_vocab = {}
    vocab_list = vocab.keys()
    for key in vocab_list:
        index = vocab[key][0]
        reverse_vocab[index] = key
    return reverse_vocab


def compute_scores(file, vocab, embeddings, threshold):
    task_name = None
    score = 0
    count = 0
    with open(file, 'r') as f:
        lines = f.read().split("\n")
        for line in lines:
            if len(line) == 0:
                continue
            if line[0] == ":":
                name = line.split(": ")[1]
                # print(name)
                if task_name is not None and count > 0:
                    print(task_name + "," + str(score/count) + "," + str(count))
                    score = 0
                    count = 0
                task_name = name
                continue
            w1, w2, w3, w4 = map(lambda x: x.strip().lower(), line.split(" "))
            w1 = w1.strip()
            if w1 not in vocab or (threshold is not None and vocab[w1][1] < threshold):
                continue
            if w2 not in vocab or (threshold is not None and vocab[w2][1] < threshold):
                continue
            if w3 not in vocab or (threshold is not None and vocab[w3][1] < threshold):
                continue
            if w4 not in vocab or (threshold is not None and vocab[w4][1] < threshold):
                continue
            v1 = embeddings[vocab[w1][0]]
            v2 = embeddings[vocab[w2][0]]
            v3 = embeddings[vocab[w3][0]]
            v4 = embeddings[vocab[w4][0]]

            res = 1 - spatial.distance.cosine(v2 - v1, v4 - v3)
            score += res
            count += 1
        if count > 0:
            print(task_name + "," + str(score / count) + "," + str(count))


if __name__ == "__main__":

    config = Config('config.ini')
    vocab, _, _ = get_vocabulary(config)

    threshold = None
    if len(sys.argv) > 2:
        threshold = int(sys.argv[2])

    embedding_file_name = sys.argv[1]
    embeddings = np.loadtxt(embedding_file_name, delimiter=",")
    compute_scores(config.analogy_dataset, vocab, embeddings, threshold)


