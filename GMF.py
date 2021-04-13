import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Lambda, Activation, multiply, concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path',
                        nargs='?',
                        default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset',
                        nargs='?',
                        default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size.')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=128,
                        help='Embedding size.')
    parser.add_argument(
        '--num_neg',
        type=int,
        default=4,
        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate.')
    parser.add_argument(
        '--learner',
        nargs='?',
        default='adam',
        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out',
                        type=int,
                        default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def get_model(train, num_users, num_items, latent_dim):

    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)

    # Input variables
    user_input = Input(shape=(1, ), dtype='int32', name='user_input')
    item_input = Input(shape=(1, ), dtype='int32', name='item_input')

    # Multi-hot User representation and Item representation
    user_rating = Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(
        user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(
        item_input)
    user_rating = Reshape((num_items, ))(user_rating)
    item_rating = Reshape((num_users, ))(item_rating)
    #print(user_rating.shape, item_rating.shape)

    # GMF part
    userlayer = Dense(latent_dim,
                      activation="linear",
                      name='gmf_user_embedding')
    itemlayer = Dense(latent_dim,
                      activation="linear",
                      name='gmf_item_embedding')
    user_latent_vector = userlayer(user_rating)
    item_latent_vector = itemlayer(item_rating)

    # Element-wise product of user and item embeddings
    predict_vector = multiply([user_latent_vector, item_latent_vector])

    # Final prediction layer
    prediction = Dense(1,
                       activation='sigmoid',
                       kernel_initializer=initializers.lecun_normal(),
                       name='prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)

    return model


def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    tt = time()
    args = parse_args()
    latent_dim = args.latent_dim
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1  #mp.cpu_count()
    print("GMF arguments: %s" % (args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' % (args.dataset, latent_dim,
                                                   time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(train, num_users, num_items, latent_dim)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate),
                      loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate),
                      loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate),
                      loss='binary_crossentropy')

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK,
                                   evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(
            train, num_negatives)

        # Training
        hist = model.fit(
            [np.array(user_input), np.array(item_input)],  #input
            np.array(labels),  # labels 
            batch_size=batch_size,
            nb_epoch=1,
            verbose=0,
            shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives,
                                           topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(
                ndcgs).mean(), hist.history['loss'][0]
            print(
                'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %
          (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" % (model_out_file))

    print("总时长为")
    print(time() - tt)
