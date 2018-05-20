import nltk
import sklearn
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import os
import random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

torch.manual_seed(1)
# convert it to LSTMText2Word
class LSTMText2Word(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, labelset_size, lstm_layers, training_epochs, batch_size, pack_dim):
        super(LSTMText2Word, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = labelset_size
        self.lstm_layers = lstm_layers
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.lstm_layers, \
                            bias=False, batch_first=False, dropout=0, bidirectional=False)
        self.pack_dim = pack_dim
        # The linear layer that maps from hidden state space to label space
        self.hidden2label = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        
        return (autograd.Variable(torch.zeros(self.lstm_layers, self.batch_size, self.hidden_dim)),
               autograd.Variable(torch.zeros(self.lstm_layers, self.batch_size, self.hidden_dim)))
        # return (autograd.Variable(torch.zeros(self.lstm_layers, 1, self.hidden_dim).double()), 
        #        autograd.Variable(torch.zeros(self.lstm_layers, 1, self.hidden_dim).double()))

    def forward(self, sequence):
        '''
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hiddenq)
        label_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores
        '''
        # print("Print sequence and size:")
        # print(sequence.size())
        # print(sequence.view(len(sequence), 1, -1).size())
        # lstm_out, self.hidden = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden)
        # -1 means inferring from other dimensions

        # lstm_out, self.hidden = self.lstm(sequence.view(len(sequence), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm(sequence)
        unpacked, unpacked_sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # get the last timestep
        unpacked_sequence_lengths = torch.FloatTensor(unpacked_sequence_lengths)
        idx = (unpacked_sequence_lengths - torch.ones(unpacked.size(0))).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        idx = autograd.Variable(idx.long())
        decoded = unpacked.gather(1, idx).squeeze() # batch size * hidden dimension
        # lstm_last = lstm_out[-1, :, :]
        # print("Print sliced lstm output size:", lstm_last.size())
        label_space = self.hidden2label(decoded.view(self.batch_size, -1))
        # print("Size of label space:", label_space.size())
        predicted_labels = F.log_softmax(label_space, dim=1)
        # print("Print predicted label:")
        # print(predicted_label)
        # print("Print predicted label size:")
        # print(predicted_label.size())
        return predicted_labels 

def load_json(label_to_index, path):
    '''
    This function processes Felicity's json file.
    For each id, extract a list of tuples, where each tuple consists of (body, majority_type).
    Note that "body" is a list of tokenized, lemmatized, lower-case, no-punctuation words.
    '''
    data_tuple = []
    data = json.load(open(path))
    for id in data:
        if data[id]['body'] != None and data[id]['majority_type'] != None:
            body = data[id]['body']
            # body = text_to_word_list(body)
            majority_type = data[id]['majority_type']
            majority_type = label_to_index[majority_type]
            tup = tuple([body, majority_type])
            data_tuple.append(tup)
    print("Data file is loaded.")
    with open('training_tuples.txt', 'w') as result:
        json.dump(data_tuple, result)
    print("The data tuples are saved as 'training_tuples.txt'.")

def text_to_word_list(text):
    '''
    This function is implemented within load_json().
    It cleans the raw comment from 'body' feature into a list 
    of tokenized, lemmatized, lower-case, no-punctuation words.
    '''
    if len(text) == 0:
        text = 'ni_zhao_bu_dao_de'
    # word_list = tokenizer.tokenize(text)
    word_list = nltk.word_tokenize(text)
    word_list = [lemmatizer.lemmatize(word.lower()) if word != 'ni_zhao_bu_dao_de' else word for word in word_list]
    return word_list

def load_glove():
    '''
    This function loads the pretrained GloVe vectors trained from CommonCrawl.
    '''
    embeddings = {}
    cmc_fname = 'glove.840B.300d.txt'
    wiki_fname = 'glove.6B.300d.txt'
    glove_name = 'glove_wiki.txt'
    DIR_PATH = '/home/zsong/working/data/glove'
    with open(os.path.join(DIR_PATH, wiki_fname)) as glove:
        for line in glove:
            values = line.split()
            # word, vector = values[0], np.asarray(values[1:], dtype='float32')
            word, vector = values[0], values[1:]
            raw_string = ''.join(vector)
            if 'com' in raw_string or '@' in raw_string:
                continue
            if '.' in vector:
                continue
            if 'name@domain.com' in vector:
                continue
            vector = [float(i) for i in vector]
            embeddings[word] = vector
    print("Gloved is loaded.")
    with open(glove_name, 'w') as result:
        json.dump(embeddings, result)
    print("Glove word vectors are saved as {}.".format(glove_name))
         
def pad_nonwords(longest_len, sorted_tokenized_sequence):
    '''
    This function pads a comment that is less than 4856 words with non-words placeholders.
    '''
    pad_counts = longest_len - len(sorted_tokenized_sequence)
    output = sorted_tokenized_sequence + ['ni_zhao_bu_dao_de'] * pad_counts
    return output

def prepare_sequence(comment, glove):
    '''
    This function converts a sequence of words into a list of word vectors.
    The output is contained in a PyTorch variable.
    '''
    word_list = comment
    # word_list = text_to_word_list(comment)
    # word_list = pad_comment(word_list) # padding
    sequence = [get_word_vectors(word, glove) for word in word_list]
    # sequence = np.array(sequence)
    # tensor = torch.from_numpy(sequence)
    tensor = torch.FloatTensor(sequence)
    # tensor = torch.DoubleTensor(sequence)
    # return autograd.Variable(tensor)
    return tensor

def prepare_label_vector(label):
    '''
    This function specifies input format.
    '''
    tensor = [0] * 9
    tensor[label] = 1
    tensor = torch.LongTensor(tensor)
    # tensor = torch.LongTensor(label)
    # return autograd.Variable(tensor)
    return tensor
    

def get_word_vectors(word, glove):
    '''
    This function retrieves the respective word vector for the word.
    If the word is not in the model, then return a Numpy array of zeros.
    Return: a word vector of 1 * 300.
    '''
    if word in glove:
        # return torch.from_numpy(np.array(glove[word]))
        return glove[word]
    else:
        # return autograd.Variable(torch.from_numpy(np.zeros(300)))
        return [0.0] * 300

def batch_processing(chunk_tuple, batch_size):
    '''
    This function takes the input and returns a packed padded sequence of data and a list of their corresponding labels.
    The input is a chunk of data tuples (sequence, label) with the size batch_size.
    '''
    strings, labels = zip(*chunk_tuple)
    word_lists = [text_to_word_list(string) for string in strings]
    tuples = zip(word_lists, labels)
    # sorted_by_length = tuples.sort(key=lambda t: len(t[0]), reverse=True)
    sorted_by_length = sorted(tuples, key=lambda t: len(t[0]), reverse=True)
    word_lists, labels = zip(*sorted_by_length)
    sequence_lengths = [len(w_list) for w_list in word_lists]
    longest_length = len(word_lists[0])
    word_lists = [pad_nonwords(longest_length, w_list) for w_list in word_lists]
    ###
    # sequence_inputs = torch.zeros((batch_size, longest_length, 300))
    sequence_inputs = [prepare_sequence(w_list, vectors) for w_list in word_lists] 
    sequence_inputs = torch.stack(sequence_inputs, dim=0)
    sequence_inputs = autograd.Variable(sequence_inputs)
    ###
    # The size of sequence inputs should be batch_size * longest_length * num_embeddings
    pack = nn.utils.rnn.pack_padded_sequence(sequence_inputs, sequence_lengths, batch_first=True)
    # labels = [prepare_label_vector(label) for label in labels]
    # labels = autograd.Variable(torch.stack(labels, dim=0))
    labels = autograd.Variable(torch.LongTensor(labels))
    return pack, labels

def clean_data(data_tuple):
    cleaned_list = []
    for idx in range(len(data_tuple)):
        dt = data_tuple[idx]
        if len(dt[0]) != 0 and not dt[0].isspace():
            cleaned_list.append(dt)
    random.Random(0).shuffle(cleaned_list)
    with open('training_tuples.txt', 'w') as result:
        json.dump(cleaned_list, result)
    print("Data is and cleaned and shuffled.")

def training(train_data, vectors):
    '''
    This function replicates the training process below.
    There is no return value. The model will be saved with torch.save() and later loaded with torch.load(). 
    '''
    model = LSTMText2Word(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, LAYER_NUM, EPOCHS, BATCH_SIZE, PACK_DIM)
    print("Print LSTM architecture:")
    print(model)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.NLLLoss(reduce=True)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    training_epochs = model.training_epochs
    label_scores = 0
    print('Initialize label score', label_scores)
    print("Start training:")
    for epoch in range(training_epochs):
        loss = 0
        i = 0
        for idx in range(0, len(train_data), model.batch_size):
            
            print("Training example {}".format(i))
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.

            model.hidden = model.init_hidden()
            # Step 2. Get our inputs ready for the network.
            chunks = train_data[idx : idx + model.batch_size]
            sequence_pack, labels = batch_processing(chunks, len(chunks)) 
            # Step 3. Run our forward pass.
            # print(model(sequence_input))
            # print((model(sequence_input)).size())
            predicted_labels = model(sequence_pack)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # temp = nn.LogSoftmax(predicted_labels)
            # print('Print first element of predicted_labels:', predicted_labels[0])
            # print('Print first element of labels:', labels[0])
            # print('The size of predicted_labels:', predicted_labels.size())
            # print('The size of labels:', labels.size())
            
            loss = loss_function(predicted_labels, labels) # need to convert predicted label from Variable to a label
            loss.backward()
            optimizer.step()
            i += model.batch_size
            if i % 100 == 0:
                print('\nEpoch [%d / %d], Loss: %.4f' %(epoch + 1, training_epochs, loss.data[0]))
    print("Training is done.")
    torch.save(model.state_dict(), 'model_batch_100_glove_100000_5_epochs_01_lr_qsub.pkl')
    # torch.save(model.state_dict(), 'model.pkl') 
    print("The model is saved as 'model_batch_100_glove_100000_5_epochs_01_lr_qsub.pkl'.")

def separate_data(data_tuple):
    data, labels = zip(* data_tuple)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.9, test_size=0.1, random_state=56)
    training = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    print("The data is split.")
    with open('train.txt', 'w') as out_file1:
        json.dump(training, out_file1)
    with open('test.txt', 'w') as out_file2:
        json.dump(test, out_file2)
    print("The training and test sets are saved as 'train.txt' and 'test.txt'.")

def testing(test_data, vectors):
    # model = torch.load('model_glove_5000.pkl')
    model = LSTMText2Word(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, LAYER_NUM, EPOCHS, BATCH_SIZE, PACK_DIM)
    model.load_state_dict(torch.load('lol.pkl'))
    correct = 0
    total_loss = 0
    total = 0
    data_size = 100000
    loss_function = nn.NLLLoss()
    ntokens = model.output_dim
    # prediction_result = []
    i = 0
    for idx in range(0, len(test_data), model.batch_size):
        chunks = test_data[idx : idx + model.batch_size]
        sequence_pack, labels = batch_processing(chunks, model.batch_size)
        predicted_labels = model(sequence_pack)
        ###
        # total_loss += data_size * loss_function(predicted_labels, labels)
        # total += labels.size(0)
        ###
        _, prediction = torch.max(predicted_labels.data, 1)
        total += labels.size(0)
        correct += prediction.eq(labels.data.view_as(prediction)).sum()
        # print('The prediction: {}, Number of correct predictions: {}, Number of test examples: {}'.format(prediction, correct, total))
        ###
        '''
        _, prediction = torch.max(predicted_labels.data, 1)
        total += labels.size(0)
        correct += prediction.eq(labels.data.view_as(prediction)).sum()
        '''
        loss = loss_function(predicted_labels, labels)
        i += model.batch_size
        if i % 1000 == 0:                 
            print('Loss: %.4f' %(loss.data[0]))
            # print('Loss: %.4f' %(total_loss.data[0]))
    # print('Accuracy of model trained on 100000 comments and test on 1000 comments: total_loss / total = {}'.format(1.0 * total_loss / total))
    print("Getting {} correct out of {} examples".format(correct, total))
    print('Accuracy of model trained on 100000 comments and test on 1000 comments: %d %%' % (100.0 * correct / total))
    # with open('result.txt', 'w') as cf:
    #    json.dump(prediction_result, cf)


data_tuple = []
DATA_PATH = "/home/zsong/working/my_trial_DIR/post_df_parent_text_unified.json" 
vectors = []
label_to_index = {"agreement": 0, "announcement": 1, "answer": 2, "appreciation": 3, "disagreement": 4, "elaboration": 5, "humor": 6, "negativereaction": 7, "question": 8}
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
OUTPUT_DIM = len(label_to_index)
LAYER_NUM = 2
EPOCHS = 5
BATCH_SIZE = 100
PACK_DIM = 1

if __name__ == "__main__":
    # load_json(label_to_index, DATA_PATH)
    # load_glove() 
    # altogether 101488 examples
    data_tuple = json.load(open('training_tuples.txt'))
    print("Number of Training Examples: {}".format(len(data_tuple)))
    # clean_data(data_tuple)
    # data_tuple = json.load(open('training_tuples.txt'))
    # print("Number of cleaned data points: {}".format(len(data_tuple)))
    # separate_data(data_tuple)
    # train = json.load(open('train.txt'))
    # test = json.load(open('test.txt')) 
    train = data_tuple[: 100000]
    # test = data_tuple[40000: 50000][:]
    # random.Random(0).shuffle(test)
    ########################################## Loading different glove embeddings
    vectors = json.load(open('glove.txt'))
    # vectors = json.load(open('glove_wiki.txt')) # test with short corpus
    # vectors = [] # test
    ##########################################
    training(train, vectors)
    # testing(test, vectors)

