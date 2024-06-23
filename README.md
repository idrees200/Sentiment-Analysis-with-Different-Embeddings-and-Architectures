# Sentiment Analysis with Different Embeddings and Architectures

## Overview

This project explores various deep learning architectures and pre-trained embeddings for sentiment analysis on an Urdu sentiment corpus. The objective is to build models that effectively classify text sentiments as positive or negative.

## Dataset

The dataset used for this project is the Urdu Sentiment Corpus v1 (`urdu-sentiment-corpus-v1.tsv`). It consists of tweets labeled as either positive ('P') or negative ('N') sentiment.

## Models Explored

### Recurrent Neural Network (RNN)

- Simple RNN layers with depths of 2 and 3 layers and dropout rates of 0.3 and 0.7.
- Trained using Word2Vec, GloVe, FastText embeddings.

### Gated Recurrent Unit (GRU)

- GRU layers with configurations similar to RNN.
- Utilized Word2Vec, GloVe, FastText embeddings.

### Long Short-Term Memory (LSTM)

- LSTM layers with configurations similar to RNN and GRU.
- Employed Word2Vec, GloVe, FastText embeddings.

### Bidirectional LSTM (BiLSTM)

- Bidirectional LSTM layers with configurations similar to LSTM.
- Integrated Word2Vec, GloVe, FastText embeddings.

### ELMo Embeddings

- AllenNLP's ELMo embeddings for capturing contextual word representations.
- Fed into a feedforward neural network for sentiment classification.

## Pre-trained Embeddings

- **Word2Vec**: GoogleNews-vectors-negative300.bin
- **GloVe**: glove.6B.300d.txt
- **FastText**: wiki-news-300d-1M.vec

These embeddings are used to initialize the embedding layers of the models, leveraging semantic and syntactic information from large corpora.

## Evaluation Metrics

For each model configuration and embedding type, the following metrics are evaluated on the test set:

- **Accuracy**: Overall accuracy of sentiment classification.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of all actual positive instances.
- **F1-score**: Harmonic mean of precision and recall.

## Results and Analysis

- Comparison and analysis of model performance across different architectures and embeddings.
- Identification of the best-performing model configuration for Urdu sentiment analysis.
- Discussion of challenges encountered and insights gained from the experiments.

## Dependencies

Ensure the following Python libraries are installed to run the project:

- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- gensim
- allennlp

## Usage

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the models and experiments as outlined in the provided notebooks or scripts.

