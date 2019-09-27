# Contains :class: `WordGenerator` which can be based on either a
# 'statistical language model' or a 'neural language model' to generate
# words. Statistical language model comprises of the bigram language model
# and trigram language model


from pathlib import Path
from keras.preprocessing.text import one_hot
from xwordgen.neural import NeuralLanguageModel


class WordGenerator:
    """
    The :class: `WordGenerator` builds either a statistical or a neural
    language model on the Brown Corpus. Given an input sequence, it can
    be used to predict the word which follows the given sequence of words.

    Parameters:
    -----
    lm (str):
        Type of language model - either 's' (statistical) or 'n' (neural)
    input_size (int):
        Parameter for Neural Language Model, size of the input sequence
        Default = 10
    batch_size (int):
        Parameter for Neural Language Model, size of batch for fitting
        the LSTM neural model
    epochs (int):
        Parameter for Neural Language Model, number of epochs for training   
    """
    def __init__(self, lm, **kwargs):
        """
        Initializes :class: `WordGenerator`
        """
        saved_models_path = Path.cwd().parent / 'saved_models'
        if lm == 's':
            pass
        elif lm =='n':
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            parameters = {'input_size': 10, 'batch_size': 128, 'epochs': 10}
            parameters.update(kwargs)

            self.nlm = NeuralLanguageModel(params= parameters, path=saved_models_path)


    def run(self, input_sequence):
        """
        Predicts the next word, given an input sequence

        Arguments
        ---------
            input_sequence (str):
                Input sequence
        Returns
        -------
            word (str):
                Predicted word from neural language model
        """
        input_tokens = input_sequence.split()
        # Removing low frequency words from input sequence
        input_tokens = [word for word in input_tokens if word not in self.nlm.low_freq_words]
        if set(input_tokens).issubset(set(self.nlm.dictionary)):
            encoded_input = one_hot(input_tokens, len(self.dictionary))
            word = self.nlm.model.predict(input_tokens)
            return word
        else:
            print('One of words in text sequence doesn\'t exist in dictionary')
