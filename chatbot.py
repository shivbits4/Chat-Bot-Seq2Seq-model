# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:43:48 2020

@author: 91842
"""

import numpy as np
import tensorflow as tf
import re
import time

########## Data pre-processing ##############

conversations =  open("cornell movie-dialogs corpus/movie_conversations.txt", encoding='utf-8', errors = 'ignore').read().split('\n')
lines = open("cornell movie-dialogs corpus/movie_lines.txt", encoding='utf-8', errors = 'ignore').read().split('\n')



id2line = dict((x.split(' +++$+++ ')[0], x.split(' +++$+++ ')[4]) for x in lines if len(x.split(' +++$+++ '))==5 )

list_conversation = [x.split(' +++$+++ ')[3][1:-1].replace("'","").replace(" ","").split(",") for x in conversations if len(x.split(' +++$+++ ')) == 4];


questions = []
answers = []
for list in list_conversation:
    for i in range(0,len(list)-1):
        questions.append(id2line[list[i]])
        answers.append(id2line[list[i+1]])


def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
    return text

clean_questions = []
clean_answers = []
for i,j in zip(questions, answers):
    text = cleanText(i)
    clean_questions.append(text)   
    text = cleanText(j)
    clean_answers.append(text)


word2count = {}

for x,y in zip(clean_questions,clean_answers):
    for word in x.split(" "):
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] +=1
    
    for word in y.split(" "):
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] +=1
        

threshold = 20

questionswords2int = {}
answerswords2int = {}
word_number = 0;
for word,count in word2count.items():
    if count >=threshold:
        questionswords2int[word] = word_number
        answerswords2int[word] = word_number
        word_number += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = word_number
    answerswords2int[token] = word_number
    word_number += 1
    
answersint2words = {count:word for word,count in answerswords2int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += " " +tokens[1]
    

question_vector = []
for line in clean_questions:
    question = []
    for x in line.split(" "):
        if x not in questionswords2int:
            question.append(questionswords2int['<OUT>'])
        else:
            question.append(questionswords2int[x])
    question_vector.append(question)
        


answer_vector = []
for line in clean_answers:
    question = []
    for x in line.split(" "):
        if x not in questionswords2int:
            question.append(answerswords2int['<OUT>'])
        else:
            question.append(answerswords2int[x])
    answer_vector.append(question)


# Sorting question and answers by length for better performance
sorted_questions_to_int = []
sorted_answers_to_int = []

for length in range(1,25+1):
    for x in enumerate(question_vector):
        if len(x[1]) == length:
            sorted_questions_to_int.append(question_vector[x[0]])
            sorted_answers_to_int.append(answer_vector[x[0]])
        

############ Building Seq2Seq Moddel ####################

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'inputs')
    targets = tf.placeholder(tf.int32, [None,None], name = 'targets')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs,targets,lr,keep_prob


# Preprocessing the targets
# For targets to be accepted at the decoder, the targets must have a specific format
# The target must be in the batches, becauses RNN doesn't take single sequence
# The targets must have a SOS token at the start of every vector
# This format is necessary for creating the embeddings for our decoding layer
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>']) # Creates a tensor of shape [batch_size,1] and value = SOS 
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    # preprocessed_targets = decoder input
    return preprocessed_targets

# Creating the Encoder RNN layer
# rnn_inputs = 
# rnn_size = Number of input tensors of the encoder rnn layer / Number of neurons per layer
# sequence_length = length of each question in the batch
# num_layers = as the name suggests, it is the number of layers
# keep_prob = Dropout params
# sequence_length = list of the length of each question in the batch
# We return only the encoder’s state because it is the input for our decoding layer.
def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob, sequence_length):
    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
    
    # Drop out is deactivating a certain percentage of neurons while training iteration, Therfore neurons weights are not updated
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    # After the below line, encoder_cell will have num_layers layers of lstm_dropout [1]*3 = [1,1,1]
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output ,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw= encoder_cell,
                                                      sequence_length= sequence_length, inputs= rnn_inputs, 
                                                      dtype = tf.float32)
    
    # encoder_state = input for the decoding layer
    return encoder_state 

#  attention_keys = keys to be compared with the target states
#  attention_values = values used to construct a context vector
#  attention_score_function = to get similarities between keys and the target states
#  attention_construct_function = used to built the attention states
def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input, sequence_length, decoding_scope,
                        output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option = 'bahdanau', 
                                                                                                                                    num_units = decoder_cell.output_size)
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],attention_keys, attention_values, attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
                                                                       decoder_cell, training_decoder_function,decoder_embedded_input,
                                                                       sequence_length, 
                                                                       scope = decoding_scope)
    decoder_output_dropout = tf.nn.Dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)
    

# Decoding the test/ Validation test

def decode_test_set(encoder_state,decoder_cell,decoder_embedded_input, sos_id, eos_id, maximum_length, num_words , decoding_scope,
                        output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option = 'bahdanau', 
                                                                                                                                    num_units = decoder_cell.output_size)
    
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,encoder_state[0],
                                                                              attention_keys, 
                                                                              attention_values, attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embedded_input, sos_id, 
                                                                              eos_id, maximum_length, num_words,
                                                                              name = "attn_dec_inf")
    
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
                                                                       decoder_cell, test_decoder_function, 
                                                                       scope = decoding_scope)
    
    return test_predictions

# Creating decoder Rnn
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix,
                encoder_state, num_words,
                sequence_length, rnn_size,
                num_layers, word2int,
                keep_prob, batch_size):
    
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weight = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer() 
        output_function = lambda x: tf.contrib.layers.fully_connected(x,num_words,
                                                                     None,
                                                                     scope = decoding_scope,
                                                                     weights_initializer = weight,
                                                                     biases_initializer = biases)
        
        training_predictions = decode_training_set(encoder_state, decoder_cell,decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int["<SOS>"], 
                                           word2int["<EOS"],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,keep_prob,batch_size)
        
        return training_predictions, test_predictions
        

# Building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words,
                  questions_num_words, encoder_embedding_size,decode_embedding_size, rnn_size,
                  num_layers, questionswords2int):
    
    # This is the input of the encoder
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words+1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0,1))
    
    # Output of the encoder. encoder_state is the input of the decoder layer
    encoder_state = encoder_rnn_layer(encoder_embedded_input,rnn_size, num_layers,keep_prob,sequence_length)
    
    # Answers to the questions
    preprocessed_targets = preprocess_targets(targets,questionswords2int,batch_size)
    
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decode_embedding_size],0,1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,
                                                         encoder_state,questions_num_words,sequence_length,rnn_size,
                                                         num_layers,questionswords2int, keep_prob, batch_size)
    
    return training_predictions, test_predictions


# Setting Hyperparameters

epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoder_embedding_size = 512
decode_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
# Always reset the graph before opening the session.But Why?
tf.reset_default_graph()
session = tf.InteractiveSession()

# Load the inputs of our seq2seq model
inputs,targets,lr,keep_prob = model_inputs()

# Setting sequence_length
sequence_length = tf.placeholder_with_default(25,None, name = "sequence_length")

# Getting the shape of the input tensor
input_shapes = tf.shape(inputs)

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,keep_prob,batch_size,sequence_length,
                                                       len(answerswords2int), len(questionswords2int),
                                                       encoder_embedding_size,decode_embedding_size,
                                                       rnn_size,num_layers,questionswords2int)

# Settings loss error, optimizer, and Gradient clipping

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets,
                                                  tf.ones([input_shapes[0], sequence_length]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradiets = [ (tf.clip_by_value(grad_tensor, -5.0, 5.0). grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradiets)

def apply_padding(batch_of_sequence, word2int):
    max_sequence_length  = max([len(sequence) for sequence in batch_of_sequence])
    return [sequence + [word2int['<PAD>']]*(max_sequence_length - len(sequence)) for sequence in batch_of_sequence]


# Split the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    # double slash is used to get integer value 
     for batch_index in range(0, len(questions)//batch_size):
         start_index = batch_index*batch_size
         questions_in_batch = questions[start_index:start_index+batch_size]
         answers_in_batch = answers[start_index:start_index + batch_size]
         padded_questions_in_batch = apply_padding(questions_in_batch, questionswords2int)
         padded_answers_in_batch = apply_padding(answers_in_batch, answerswords2int)
         yield padded_questions_in_batch, padded_answers_in_batch
         

# Splitting the questions and answers into trainingand validation datasets
training_validation_split = int(len(sorted_questions_to_int)*0.15)
training_questions = sorted_questions_to_int[training_validation_split:,:]
validation_questions = sorted_questions_to_int[0:training_validation_split,:]
training_answers = sorted_answers_to_int[training_validation_split:,:]
validation_answers = sorted_answers_to_int[0:training_validation_split,:]


# Training
# We will check the training loos at every 100 batches
batch_index_check_training_loss = 100

# 
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) -1
total_training_loss_error = 0

# To store loss the at every batch
list_validation_loss_error = []
# If the validation loss didnt improve at every step then we will increase the count. Why?  
early_stopping_check = 0

# Once early_stopping_check reaches a particular number then we stop everything
early_stopping_stop = 1000

# TO store the weight of the trained chat bots. Below is the file path that will contain the weights
checkpoint = "chatbot_weights.ckpt"

session.run(tf.global_variables_initializer())

for epoch in range(1, epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _ ,batch_training_loss_error =  session.run({optimizer_gradient_clipping, loss_error}, {inputs: padded_questions_in_batch,
                                                                targets: padded_answers_in_batch,
                                                                lr: learning_rate,
                                                                sequence_length: padded_answers_in_batch.shape[1],
                                                                keep_prob: keep_probability})
    
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
    
        if batch_index % batch_index_check_training_loss ==0 :
            print("Epoch: {:>3} /{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 batches: {:d} seconds".format(epoch,
                                                                                                                                    epochs,
                                                                                                                                    batch_index,
                                                                                                                                    len(training_questions)// batch_size,
                                                                                                                                    total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                    int(batch_time*batch_index_check_training_loss)))
            total_training_loss_error = 0
        
        if batch_index % batch_index_check_validation_loss ==0 and batch_index > 0:
            total_validation_loss_error =  0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error =  session.run( loss_error, {inputs: padded_questions_in_batch,
                                                                targets: padded_answers_in_batch,
                                                                lr: learning_rate,
                                                                sequence_length: padded_answers_in_batch.shape[1],
                                                                keep_prob: 1})
    
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            
            average_validation_loss_error = total_validation_loss_error/ (len(validation_questions)/batch_size)
            print("Validation Loss Error: {:>6.3f}, batch Validatio Time: {:d}seconds".format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Do not speak better! I need to practise more")
                early_stopping_check +=1
                if early_stopping_check == early_stopping_stop:
                    break
                
    if early_stopping_check == early_stopping_stop:
        print("Cannot speak better")
        break

print("Game over")