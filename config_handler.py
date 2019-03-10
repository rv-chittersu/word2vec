import configparser
import os


class Config:

    project_path = '.'

    files_split_sourcefile = None
    vocabulary_sourcefile = None
    results_folder = None
    simlex_dataset = None
    analogy_dataset = None

    embedding_dimensions = 60
    negative_samples = 120
    epochs = 3
    learning_rate = 0.5
    window_size = 5

    def __init__(self, file):
        # parse the config file
        config = configparser.ConfigParser()
        if file is None:
            file = 'config.ini'
        config.read(file)

        self.project_path = config.get('LOCATIONS', 'ProjectLocation', fallback='.')

        self.files_split_sourcefile = self.project_path + '/' + config.get('LOCATIONS', 'DataFileNamesSplit',
                                                                           fallback='data/default.split.txt')
        self.vocabulary_sourcefile = self.project_path + '/' + config.get('LOCATIONS', 'VocabularySource',
                                                                          fallback='data/default.vocab')
        self.analogy_dataset = self.project_path + '/' + config.get('LOCATIONS', 'AnalogyDataset',
                                                                   fallback='data/questions-words.txt')
        self.simlex_dataset = self.project_path + '/' + config.get('LOCATIONS', 'SimLexDataset',
                                                                    fallback='data/SimLex-999.txt')
        self.results_folder = self.project_path + '/' + config.get('LOCATIONS', 'ResultsDirectory',
                                                                   fallback='results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.embedding_dimensions = config.getint('MODEL_PARAMETERS', 'EmbeddingDimensions', fallback=60)
        self.negative_samples = config.getint('MODEL_PARAMETERS', 'NegativeSamples', fallback=120)
        self.epochs = config.getint('MODEL_PARAMETERS', 'Epochs', fallback=3)
        self.learning_rate = config.getfloat('MODEL_PARAMETERS', 'LearningRate', fallback=0.5)
        self.window_size = config.getint('MODEL_PARAMETERS', 'WindowSize', fallback=5)

    def __str__(self):
        return "------CONFIG-------\n"\
               + "project-path - " + self.project_path + "\n"\
               + "data-split-info - " + self.files_split_sourcefile + "\n"\
               + "source-vocabulary - " + self.vocabulary_sourcefile + "\n" \
               + "simlex-dataset - " + self.simlex_dataset + "\n" \
               + "analogy-dataset - " + self.analogy_dataset + "\n" \
               + "embedding-dimensions - " + str(self.embedding_dimensions) + "\n"\
               + "negative-samples - " + str(self.negative_samples) + "\n"\
               + "epochs - " + str(self.epochs) + "\n"\
               + "learning-rate - " + str(self.learning_rate) + "\n"\
               + "window-size - " + str(self.window_size) + "\n\n"
