import numpy as np
from flaskapp_utils import *
import emoji
import csv

## Load training data
X_train, Y_train = read_csv('data/flask_TrainingData_orig.csv')     # Use windows comma separated CSV format

## Data prep
maxLen = len(max(X_train, key=len).split())

## Load pretrained GloVe embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')




# Custom Functions

def sentence_to_avg(sentence, word_to_vec_map):
    """
        Converts a sentence (one string) into a list of words (many strings), then averages the GloVe representations of each word into one composite sentence vector. This captures the sentence's sentiment and represents it as one vector.

        Arguments:
        sentence -- string, one training example from X
        word_to_vec_map -- GloVe dictionary mapping every word in a vocabulary into its n-dimensional vector representation (we use 50D; for more information, see: https://nlp.stanford.edu/projects/glove/)

        Returns:
        avg -- the final vector, representing the averaged GloVe representations (numpy-array of shape (50,))
        """

    # Split sentence into list of lower case words
    words = sentence.lower().split()

    # Initialize the average word vector (must have the same shape as the word vectors, and we are using the 50-dimensional representations)
    avg = np.zeros(50,)

    # Average the word vectors. Loop over "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg/len(words)
    return avg


# Create a simple baseline model to determine low end of performance (model can be improved, for example, by incorporaing an n-layer LSTM)
def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
        Model to train word vector representations in numpy.

        Arguments:
        X -- input data, numpy array of sentences as strings, of shape (m, 1)
        Y -- labels, numpy-array of integers (between 0 and total number of classes), numpy-array of shape (m, 1)
        word_to_vec_map -- GloVe dictionary mapping every word in a vocabulary into its n-dimensional vector representation (we use 50D; for more information, see: https://nlp.stanford.edu/projects/glove/)
        learning_rate -- learning rate for stochastic gradient descent
        num_iterations -- number of iterations

        Returns:
        pred -- vector of predictions, numpy-array of shape (m, 1)
        W -- weight matrix of the softmax layer, of shape (n_y, n_h)
        b -- bias of the softmax layer, of shape (n_y,)
        """

    # Define number of training examples, number of classes, and GloVe vector dimensionality
    m = Y.shape[0]                          # Number of training examples
    n_y = 5                                 # Define number of classes the model predicts
    n_h = 50                                # Number of dimensions the GloVe vectors have

    # Initialize parameters (Xavier initialization used)
    # From https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference: He initialization works better for layers with ReLu activation. Xavier initialization works better for layers with sigmoid activation.
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y)

    # Optimization loop (vectorized as much as is practicable)
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples

            # Averaged word vectors of words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # avg forward propagated through the softmax layer
            z = np.dot(W,avg)+b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (capital A indicates whole output of the softmax(Z) (softmax on all input examples), not just each output "a" for softmax(z).)
            cost = -1*np.sum(Y_oh[i]*np.log10(a))

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters using Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            pred = predict(X, W, b, word_to_vec_map)

    return pred, W, b


def sentences_to_indices(X, word_to_index, max_len):
    """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape is fed to `Embedding()

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
        max_len -- maximum number of words in a sentence. Sentences in X must be shorter than max_len

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """

    m = X.shape[0]                                   # number of training examples

    # Initialize X_indices
    X_indices = np.zeros((m, max_len))

    for i in range(m):                               # loop over training examples (i, j, k indexing convention used)

        # Convert the ith training sentence to lower case, then split into a list of words.
        sentence_words = X[i].lower().split()

        # Loop over the words in sentence_words
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1

    return X_indices




# Make Predictions
pred, W, b = model(X_train, Y_train, word_to_vec_map)
X_my_sentences = np.array(["That play was fucking incredible", "I have a lot of emails to send but work is going well", "I have missed you and loved you", "I hate you leave immediately", "girls like you are all bitches"])
pred = predict(X_my_sentences, W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)
