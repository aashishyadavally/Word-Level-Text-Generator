"""
Contains :class: `NeuralLanguageModel` which implements a
'neural language model' to generate words

Author
------
Aashish Yadavally
"""


import math
from pathlib import Path
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import brown
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot


class NeuralLanguageModel:
    """
    Implements LSTM based neural language model, trained on the Brown Corpus.

    Parameters
    ----------
        params (dict):
            Parameters of language model
        path (pathlib.Path):
            Path to saved model directory
    """
    def __init__(self, params, path):
        """
        Initializes :class: `NeuralLanguageModel`

        Returns
        ---------
            model (keras.models.Sequential):
                LSTM Model trained on the training sequences from Brown Corpus
        """
        self.n = params['input_size']
        model_name = f"neural_model.{self.n}"
        model_path = path / f'{model_name}.h5'

        processed_data = self.process_data()
        if model_path.exists():
            # Reading model architecture from JSON and weights from hd5
            with open(path / f'{model_name}.json', 'r') as f:
                print("Trained model for this configuration already exists.")
                print("Skipping training, loading model..")
                self.model = model_from_json(f.read())
                self.model.load_weights(path / f'{model_name}.h5')
        else:
            train, test = self.split_data(processed_data)
            # Creating sequences of length (n + 1)
            train_sequences = [train[i-self.n : i+1] for i in
                               range(self.n, len(train))]
            test_sequences = [test[i-self.n : i+1] for i in
                              range(self.n, len(test))]
            # Fitting the model
            self.model = self.LSTM()
            print(f'Training model with parameters: \n {params}')
            self.model.fit_generator(self.xy_fit_generator(train_sequences,
                                                           batch_size=params['batch_size']),
                                     steps_per_epoch=math.ceil(len(train_sequences)/params['batch_size']),
                                     epochs=params['epochs'])
            path.mkdir(exist_ok=True)
            self.model.save_weights(path / f'{model_name}.h5')
            # Save the model architecture
            with open(path / f'{model_name}.json', 'w') as f:
                f.write(self.model.to_json())


    def process_data(self):
        """
        Processes the given data to remove low frequency words, one-hot encode
        the corpus

        Returns
        -------
            one_hot_text (list):
                Corpus containing one-hot encoded words
        """
        nltk.download('brown')
        word_list = brown.words()
        word_freq = dict(Counter([word.lower() for word in word_list]))
        # Computing words with frequency <=3
        self.low_freq_words = {word: freq for word, freq in
                               word_freq.items() if freq <= 3}.keys()
        # Removing low frequency words
        relevant_words = [word for word in word_list if word not in
                          self.low_freq_words]
        relevant_text = " ".join(relevant_words)
        self.dictionary = list(set(relevant_words))
        # One-hot encoding the text corpus
        one_hot_text = one_hot(relevant_text, len(self.dictionary))
        return one_hot_text


    def split_data(self, processed_data):
        """
        Splits the corpus with a 80-20 split-ratio, so as to return
        train-test data

        Arguments
        ---------
            processed_data (list):
                List of one-hot encoded corpus

        Returns
        -------
            (tuple):
                Tuple containing one-hot encoded train set, and one-hot encoded
                test set - split for train-test is 80-20
        """
        # Using first 80% of data for training and remaining for testing
        split_index = int(0.8 * len(processed_data))
        # Returning train and test splits
        return processed_data[:split_index], processed_data[split_index+1:]


    def xy_fit_generator(self, sequences, batch_size):
        """
        Generates numpy arrays of the size of 'batch_size' containing x and y
        data from the sequences, where x is a sequence of

        Arguments
        ---------
            sequences (list):
                List of train/test one-hot encoded word sequences
            batch_size (int):
                Batch size for fitting the model in an epoch

        Yields
        ------
            x, y (tuple)
                x is array of one-hot encoded word sequences
                y is array of categorical classes of next word in dictionary
        """
        for start_index in range(0, len(sequences), batch_size):
            x = np.array([sequence[:-1] for sequence in
                          sequences[start_index: start_index + batch_size]])
            y = np.array([to_categorical(sequence[-1],
                                         num_classes=len(self.dictionary),
                                         dtype='uint16')
                          for sequence in sequences[start_index: start_index + batch_size]])
            yield (x, y)


    def LSTM(self):
        """
        Defines a Keras LSTM model.

        Returns
        -------
            model (keras.models.Sequential):
                Compiled LSTM model
        """
        model = Sequential()
        model.add(Embedding(len(self.dictionary), 128, input_length=self.n))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(self.dictionary), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary)
        return model
