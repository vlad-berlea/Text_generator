import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # tokenizer will extract all words
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)


def main():
    f = open("Frankenstein.txt", encoding="utf8").read()
    processed_inputs = tokenize_words(f)
    #print(processed_inputs)

    # map all the chars to integer for the neural network
    ch = sorted(list(set(processed_inputs)))
    ch_num = dict((c, i) for i, c in enumerate(ch))
    inp_len = len(processed_inputs)
    vb_len = len(ch)


    #make data sets of length 100
    seq_length = 100
    x_data = []
    y_data = []

    for i in range(0, inp_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
        in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the character following the in_seq
        out_seq = processed_inputs[i + seq_length]

    # construct our test data set
        x_data.append([ch_num[char] for char in in_seq])
        y_data.append(ch_num[out_seq])
    
    #Convert data set to numpy array and the values to float
    X = numpy.reshape(x_data, (len(x_data), seq_length, 1))
    X = X/float(vb_len)
    y = np_utils.to_categorical(y_data)

    #specify the sequential LTSM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    #Create checkpoint for model
    filepath = "model_weights_saved.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]
    model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)
    filename = "model_saved.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    num_to_char = dict((i, c) for i, c in enumerate(ch))
    start = numpy.random.randint(0, len(x_data) - 1)
    print(start)
    #pattern = x_data[start]
    #print("Random Seed:")
    #print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
if __name__ == "__main__":
    main()
