"""
Entry point into the package - contains main() function.
The following will help in understanding the package arguments:

    $ xwordgen --help

Author
------
Aashish Yadavally
"""


import argparse
from xwordgen.word_generator import WordGenerator


def main():
    """
    Main function - entry point for `xwordgen` package
    """
    parser = argparse.ArgumentParser(description='Word Level Text Generation')
    parser.add_argument('--lm', dest='lm', default='n', type=str,
                        required=True, choices=['n', 's'],
                        help='Type of language model -\'statistical\' or \'neural\'')
    parser.add_argument('--input_size', dest='ins', default=10, type=int,
                        help='Size of sequences')
    parser.add_argument('--batch_size', dest='bs', default=128, type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int,
                        help='Number of epochs for training')
    args = parser.parse_args()

    wg = WordGenerator(lm=args.lm, ins=args.ins, bs=args.bs, epochs=args.epochs)

    input_seq = input('Enter input sequence: ')
    while len(input_seq.split()) != args.ins:
        print(f'Length of entered sequence should be {args.ins}. Please try again!')
        input_seq = input('Enter input sequence: ')

    #word = wg.run(input_seq)
    #print(f'Predicted next word is: {word}')
