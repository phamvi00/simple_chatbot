import nltk
nltk.download('punkt')
nltk.download('wordnet')
import sys
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import Input, Model
from tensorflow.keras.optimizers import SGD
import random
from scipy import sparse
from tqdm import tqdm

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


window = 2
word_lists = []
index=0
for text in words:
    # Appending to the all text list
    index+=1
    for w in range(window):
        # Getting the context that is ahead by *window* words
        if  index+1 + w < len(words):
            word_lists.append([text] + [words[(index+w+1)]])
            # Getting the context that is behind by *window* words
        if index -w - 1 >= 0:
            word_lists.append([text] + [words[(index-w - 1)]])

print("wordlist", word_lists)

def create_unique_word_dict(text:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()

    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict

unique_word_dict = create_unique_word_dict(words)
print("unique_word_dict", unique_word_dict)

# Defining the number of features (unique words)
n_words = len(unique_word_dict)

# Getting all the unique words
words = list(unique_word_dict.keys())


# Creating the X and Y matrices using one hot encoding
X = []
Y = []

for i, word_list in tqdm(enumerate(word_lists)):
    # Getting the indices
    main_word_index = unique_word_dict.get(word_list[0])
    context_word_index = unique_word_dict.get(word_list[1])

    # Creating the placeholders
    X_row = np.zeros(n_words)
    Y_row = np.zeros(n_words)

    # One hot encoding the main word
    X_row[main_word_index] = 1

    # One hot encoding the Y matrix words
    Y_row[context_word_index] = 1

    # Appending to the main matrices
    X.append(X_row)
    Y.append(Y_row)

# Converting the matrices into an array
X = np.asarray(X)
Y = np.asarray(Y)


# Deep learning:
from keras.models import Input, Model
from keras.layers import Dense

# Defining the size of the embedding
embed_size = 2

# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
model.fit(
    x=X,
    y=Y,
    batch_size=256,
    epochs=1000
    )

# Obtaining the weights from the neural network.
# These are the so called word embeddings

# The input layer
weights = model.get_weights()[0]

# Creating a dictionary to store the embeddings in. The key is a unique word and
# the value is the numeric vector
embedding_dict = {}
for word in words:
    embedding_dict.update({
        word: weights[unique_word_dict.get(word)]
        })




# pickle.dump(words,open('words.pkl','wb'))
# pickle.dump(classes,open('classes.pkl','wb'))
#
# # initializing training data
# training = []
# output_empty = [0] * len(classes)
# for doc in embedding_dict:
#     # initializing bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = doc[0]
#     # lemmatize each word - create base word, in attempt to represent related words
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     # create our bag of words array with 1, if word match found in current pattern
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
#
#     # output is a '0' for each tag and '1' for current tag (for each pattern)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#
#     training.append([bag, output_row])
#
#
# # shuffle our features and turn into np.array
# random.shuffle(training)
# training = np.array(training)
# # create train and test lists. X - patterns, Y - intents
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print("Training data created")
#
#
# # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
#
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# #fitting and saving the model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)
#
# print("model created")
