from utils import *
from model import *
import numpy as np
import random, string


if __name__ == '__main__':

    config = Config('config.ini')
    print(config)

    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))  # generate random string

    vocabulary, v_size, v_count = get_vocabulary(config)

    model = Model(config, v_size)

    print("Initializing DataHandlers")
    train, validation, test = get_document_names(config)
    train_data_handler = DataHandler(vocabulary, v_size, v_count, config, train)
    validation_data_handler = DataHandler(vocabulary, v_size, v_count, config, validation)
    test_data_handler = DataHandler(vocabulary, v_size, v_count, config, test)
    print("Initialized DataHandlers")

    training_stats = []
    files = []
    for i in range(0, config.epochs):
        print("==========================")
        print("Starting epoch:" + str(i))
        print("==========================")

        total_training_loss, training_samples = model.run(train_data_handler, "train")
        print(">> Average Training Loss over " + str(training_samples) + " samples - " +
              str(total_training_loss/training_samples))

        total_validation_loss, validation_samples = model.run(validation_data_handler, "validation")
        print(">> Average Validation Loss over " + str(validation_samples) + " samples - " +
              str(total_validation_loss/validation_samples))

        train_data_handler.reset()
        validation_data_handler.reset()

        embeddings = model.get_embeddings()
        embedding_file_name = config.results_folder + '/' + key + ".embedding.epoch-" + str(i) + '.out'
        np.savetxt(embedding_file_name, embeddings, delimiter=',')

        training_stats.append((total_training_loss/training_samples, total_validation_loss/validation_samples))
        files.append(embedding_file_name)

    test_loss, test_samples = model.run(test_data_handler, "test")
    print(">> Average Test Loss over " + str(test_samples) + "samples - " + str(test_loss / test_samples))

    model.end_session()

    result = compile_results(config, training_stats, test_loss/test_samples, files)
    print(result)
    with open(config.results_folder + '/' + key + ".results.txt", "w") as f:
        f.write(result)

