# word2vec skipgram Model
This work is done as part of [assignment](https://sites.google.com/site/2019e1246/schedule/assignment-1) for [E1 246: Natural Language Understanding (2019)](https://sites.google.com/site/2019e1246/basics). The report fot the same can be found [here]()

## Data
NLTK's Reuters dataset is used to train the skipgram model.

## File Structure
Project layout
```
data/
- default.split.txt
- default.vocab
- SimLex-999.txt
results/
config.ini
initialize.py
driver.py
model.py
data_handler.py
config_handler.py
utils.py
similar_words.py
simlex_results.py
```

### Data

This folder holds pre-generated data which is necessary for training the model. 

*SimLex-999.txt* is used for evaluation and is downloaded from [here](https://fh295.github.io/simlex.html).

*default.split.txt* contains file ids split for train validation and testing.

*default.vocab* contains vocabulary generated on training data.

*Note:* New training splits and vocabulary can be generated from *initialize.py*

### Code

*initialize.py* downloads nltk corpus. With arguments it can be used to generate training-validation split and vocabulary

*driver.py* contains main function for training the model.

*model.py* holds logic for word2vec skipgram model

*data_handler.py* is for generating inputs for model from corpus.

*config_handler.py* is interface for reading config file

*utils.py* has implementation of functions used by other files.

*similar_words.py* generates similar words for a given word based on embeddings

*simlex_results.py* calculates co-relation between simlex-999 scores and model's similarity score.

## Config File

```
[LOCATIONS]
ProjectLocation = .
VocabularySource = data/default.vocab
DataFileNamesSplit = data/default.split.txt
ResultsDirectory = results

[MODEL_PARAMETERS]
Epochs = 3
LearningRate = 0.5
EmbeddingDimensions = 60
NegativeSamples = 120
WindowSize = 5
```

## How to Run

### Run on available dataset
verify that files in config are present and run following commands from project folder
```
python initialize.py
python driver.py
``` 

### Generate new training split
*filename1* - holds file ids split for training validation and test
*filename2* - has vocabulary generated from train set.
```
python initialize.py filename1 filename2
```

update file names in *config.ini* as following
```
VocabularySource = filename2
DataFileNamesSplit = filename1
```

and run
```
python driver.py
```


## Evaluation

During training and testing is done the embeddings at each epoch are written into file. The files will be *results* folder and file names will be displayed as part of console output.

### Finding k similar words

*param1* - embedding file

*param2* - token

*param3* - output count

```
python similar_words.py embedding.out oil 10
```

### Co-relation score

*param1* - embedding file

*param2* - parts of speech (N/V/A)

*param3* - threshold (minimum occurrences in training set) -optional
```
python simlex_results.py results/emb.out N 100
```