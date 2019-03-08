from utils import *
import sys
import nltk


if __name__ == '__main__':

    print("Downloading nltk corpus")
    nltk.download("punkt")
    nltk.download("reuters")
    nltk.download("stopwords")
    print("Download successful")

    if len(sys.argv) < 3:
        exit()

    split_info_file = sys.argv[1]
    vocab_file = sys.argv[2]

    train_ids, validation_ids, test_ids = split_corpus()

    with open(split_info_file, "w") as f:
        f.write(",".join(train_ids))
        f.write("\n")
        f.write(",".join(validation_ids))
        f.write("\n")
        f.write(",".join(test_ids))

    vocab, _, _ = create_vocabulary(train_ids)

    save_vocabulary(vocab, vocab_file)

    print("Created files")
    print("Update config.ini file before running main program")
