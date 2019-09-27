# Word Level Text Generation using Neural and Statistical Language Models

## Synopsis
The goal of this project is to develop a word-level text generator, built using on top of a neural language modeler or a statistical language modeler.

The user can either select one of the saved models, or train one on his/her own on the Brown Corpus, so as to use `xwordgen` to generate words that follow a given input sequence of words.

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup
This project can easily be set up with all the prerequisite packages by following these instructions:
  1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) using the `conda_install.sh` file, with the command: `$ bash conda_install.sh`
  2. Create a conda environment from the included `environment.yml` file using the following command:
     
     `$ conda env create -f environment.yml`
  3. Activate the environment
     
     `$ conda activate xwordgen`

### Usage
 The user can get a description of the options by using the command: `$ xwordgen --help`.
 
 Furthermore, using either the saved models or the trained ones, the user will be prompted to enter an input sequence, for  which the word that follows will be generated.

### Built With
* [Python](https://www.python.org/)

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* We use [Google-Style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to document the Python modules in this project.

If you see something that could be improved, send a pull request! 
I am always happy to look at improvements, to ensure that `xwordgen`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/aashishyadavally/Word-Level-Text-Generator/blob/master/LICENSE) file for more details.
