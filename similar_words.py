from driver import *
import sys

embed = np.loadtxt("emb.1.w_5.stop_words.neg_120.out", delimiter=",")
vocab, _, _ = load_vocabulary('default.stop.vocab')

reverse_vocab = {}
vocab_list = vocab.keys()
for key in vocab_list:
    index = vocab[key][0]
    reverse_vocab[index] = key

word = sys.argv[1]
word_id = vocab[word][0]
word_emb = embed[word_id]
sim = []

print(word_emb)

for i in embed:
    sim.append(np.dot(word_emb, i))

res = [i[0] for i in sorted(enumerate(sim), key=lambda x:-1*x[1])]

for i in range(0, 30):
    print(reverse_vocab[res[i]])
