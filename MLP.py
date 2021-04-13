import numpy as np
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Lambda, Reshape, BatchNormalization, multiply, Activation, Dropout, Add
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
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
    parser.add_argument(
        '--layers',
        nargs='?',
        default='[1024,512,256,128]',
        help="Size of each layer. Note that the first layer is the "
        "concatenation of user and item embeddings. So layers[0]/2 is the embedding size."
    )
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


def get_model(train, num_users, num_items, layers):
    num_layer = len(layers)  # Number of layers in the MLP
    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)
    # Input variables
    user_input = Input(shape=(1, ), dtype='int32', name='user_input')
    item_input = Input(shape=(1, ), dtype='int32', name='item_input')

    user_rating = Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(
        user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(
        item_input)
    user_rating = Reshape((num_items, ))(user_rating)
    item_rating = Reshape((num_users, ))(item_rating)

    MLP_Embedding_User = Dense(layers[0] // 2,
                               activation="linear",
                               name='mlp_user_embedding')
    MLP_Embedding_Item = Dense(layers[0] // 2,
                               activation="linear",
                               name='mlp_item_embedding')
    user_latent = MLP_Embedding_User(user_rating)
    item_latent = MLP_Embedding_Item(item_rating)

    # The 0-th layer is the concatenation of embedding layers
    # attention part start
    vector = concatenate([user_latent, item_latent])
    attention_probs = Dense(layers[0],
                            activation='softmax',
                            name='attention_vec')(vector)
    attention_mul = multiply([vector, attention_probs])
    mlp_vector = concatenate([attention_mul, vector])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu', name='layer%d' % idx)
        mlp_vector = layer(mlp_vector)

    # Final prediction layer，得到的是一个0-1的值，即预测正确用户喜欢item的概率（推荐成功）
    prediction = Dense(1,
                       activation='sigmoid',
                       kernel_initializer=initializers.lecun_normal(),
                       name='prediction')(mlp_vector)

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
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%d.h5' % (args.dataset, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(train, num_users, num_items, layers)
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

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK,
                                   evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(
            train, num_negatives)

        # Training
        hist = model.fit(
            [np.array(user_input), np.array(item_input)],  # input
            np.array(labels),  # labels
            batch_size=batch_size,
            epochs=1,
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
        print("The best MLP model is saved to %s" % model_out_file)

    print("总时长为")
    print(time() - tt)
