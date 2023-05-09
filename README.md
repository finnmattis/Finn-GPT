# FinnGPT

The World's Worst Chatbot :(

## Description

Built with pytorch, FinnGPT is a custom-implemented encode-decode transformer that performs sequence-to-sequence tasks. It also comes with a custom-implemented byte pair encoder. It is trained off the AmbigQA dataset which can be found here: https://nlp.cs.washington.edu/ambigqa.

## Usage

As this project is in development, you can just run the file "main.py" with a python interpreter of your choice.

## Code Overview

### main.py 

The file "main.py" contains code the code that tokenizes the data, optimizes it using torch's adamW, and samples from the model.

### tokens.py

The file "tokens.py" houses the "Tokenizer" class uses Byte Pair encoding.

### transformer.py

The file "transformer.py" implements a myriad of classes, but the "Transformer" class is the one used by "main.py".

## License

This project is licensed under the MIT License - see the LICENSE file for details.