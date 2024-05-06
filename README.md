# Capstone Project: Multiword Expression Extraction
Spring 2024

## About
Multiword expressions are phrases that behave as a single semantic or syntactic unit. 
Successfully identifying them can improve the performance of downstream tasks such as parsing and machine translation. 
The goal of this project is to train a model to extract multiword expressions using a small, custom-annotated dataset. 
I experiment with encoder and decoder large language models and find the relatively small fine-tuned models to 
outperform the enormous, generic models.

For more details, please see the report or run the Streamlit application.

## Runtime Instructions
To run the application, first download 
[the models that are too large to put on GitHub](https://drive.google.com/drive/folders/11vmtYA9rQZ487ItbOcNH_yWv089q3KKb?usp=drive_link) 
and save them in the repository in a directory called saved_models.
Then, run the following in the terminal:
```
$ pip install -r requirements.txt
$ streamlit run app.py
```
The application will open in your browser.

## Code Description

* `preprocess.py`: Builds the dataset by cleaning, saving, tokenizing, and splitting the data and formatting the labels.
* `model.py`: Defines BERT and BERT-CRF models.
* `train.py`: Trains and evaluates a model with the given configuration.
* `utils.py`: Contains helper functions for pre- and post-processing.
* `prompt.py`: Prompts GPT 3.5 Turbo and stores the responses. Requires OpenAI API key.
* `postprocess.py`: Parses and evaluates the GPT responses.
* `inference.py`: Evaluates saved models and ensemble systems based on saved models.
* `process_new_data.py`: Predicts multiword expressions in unannotated data.
* `tests.py`: Performs unit tests.
* `app.py`: Builds a Streamlit application.
