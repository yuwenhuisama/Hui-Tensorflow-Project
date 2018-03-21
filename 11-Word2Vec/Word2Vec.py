'''
File: \Word2Vec.py
Project: 11-Word2Vec
Created Date: Tuesday March 20th 2018
Author: Huisama
-----
Last Modified: Tuesday March 20th 2018 10:03:42 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

URL = 'http://mattmahoney.net/dc/'
'''
    Download dataset
'''
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(URL + filename, filename)
    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print('found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

FILE_NAME = maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        print('file list:', f.namelist())
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

WORDS = read_data(FILE_NAME)
print('Data size', len(WORDS))

VOCABULARY_SIZE = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))

    dictionary = {}
    for word, _ in count:
        # Number every word
        dictionary[word] = len(dictionary)

    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # data:  [ 111, 32, 465, 0, ... ]
    # count: { ['UNK': 45456], ..., ('BAR', 1000), ... }
    # dict : { 'BAR': 1, ..., }
    # reverse_dictionary : { 1: 'Bar' }
    return data, count, dict, reverse_dictionary

DATA, COUNT, DICTIONARY, REVERSE_DICTIONARY = build_dataset(WORDS)

del WORDS
print('Most common words (+UNK)', COUNT[:5])
print('Sample data', DATA[:10], [REVERSE_DICTIONARY[i] for i in DATA[:10]])

data_index = 0

'''
    Generate batch as:
    TrainData: [123, 512, 5235, 123, ..., 551]
    TrainLabel: [1, 1, 1, 1, ..., 50]
'''
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # 1 batch for 1 context word, ensure every context word having same number of sample (hard to understand o(╥﹏╥)o)
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen = span)

    for _ in range(span):
        buffer.append(DATA[data_index])
        data_index = (data_index + 1) % len(DATA)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # may be some problems here when "a a a c b b b"
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j ] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(DATA[data_index])
        data_index = (data_index + 1 % len(DATA))
    return batch, labels

'''
    Do training
'''
batch, labels = generate_batch(batch_size = 8, num_skips = 2, skip_window = 1)
for i in range(8):
    print(batch[i], REVERSE_DICTIONARY[batch[i]], '->', labels[i, 0], REVERSE_DICTIONARY[labels[i, 0]])

BATCH_SIZE = 16
EMBEDDING_SIZE = 128
SKIP_WINDOW = 1
NUM_SKIPS = 2

VALID_SIZE = 16
VALID_WINDOW = 100
VALID_EXAMPLES = np.random.choice(VALID_WINDOW, VALID_SIZE, replace = False)
NUM_SAMPLED = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32, shape = [BATCH_SIZE, 1])
    valid_dataset = tf.constant(VALID_EXAMPLES, dtype = tf.int32)

    # with tf.device('/cpu:0'):
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE], stddev = 1.0 / math.sqrt(EMBEDDING_SIZE)))
    nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights = nce_weights,
        biases = nce_biases,
        labels = train_labels,
        inputs = embed,
        num_sampled = NUM_SAMPLED,
        num_classes = VOCABULARY_SIZE))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # Compute cosine similarity
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

    init = tf.global_variables_initializer()

NUM_STEPS = 100001
with tf.Session(graph = graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(NUM_STEPS):
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
        feed_dict = { train_inputs: batch_inputs, train_labels: batch_labels }

        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0

        # Pick the k-nearest words of context word to display
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(VALID_SIZE):
                valid_word = REVERSE_DICTIONARY[VALID_EXAMPLES[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1 : top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = REVERSE_DICTIONARY[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
