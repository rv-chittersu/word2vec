from utils import *
import sys


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Expected 2 arguments but found " + str(len(sys.argv) - 1))
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
