import numpy as np
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Flatten, concatenate, Dot, multiply, Reshape
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import DMF
import MLP
import GMF


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepCF.")
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
                        default=10,
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
        '--mlp_layers',
        nargs='?',
        default='[1024,512,256,128]',
        help="MLP layers. Note that the first layer is the concatenation "
        "of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--userlayers',
                        nargs='?',
                        default='[128,128,128]',
                        help="Size of each user layer")
    parser.add_argument('--itemlayers',
                        nargs='?',
                        default='[128,128,128]',
                        help="Size of each item layer")
    parser.add_argument(
        '--num_neg',
        type=int,
        default=4,
        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00001,
                        help='Learning rate.')
    parser.add_argument(
        '--learner',
        nargs='?',
        default='sgd',
        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out',
                        type=int,
                        default=1,
                        help='Whether to save the trained model.')
    parser.add_argument(
        '--gmf_pretrain',
        nargs='?',
        default='',
        help=
        'Specify the pretrain model file for GMF part. If empty, no pretrain will be used'
    )
    parser.add_argument(
        '--dmf_pretrain',
        nargs='?',
        default='',
        help=
        'Specify the pretrain model file for DMF part. If empty, no pretrain will be used'
    )
    parser.add_argument(
        '--mlp_pretrain',
        nargs='?',
        default='',
        help=
        'Specify the pretrain model file for MLP part. If empty, no pretrain will be used'
    )
    return parser.parse_args()


def get_model(train, num_users, num_items, latent_dim, userlayers, itemlayers,
              mlp_layers):

    gmf_num_layer = latent_dim  # Number of layers in the GMF
    dmf_num_layer = len(userlayers)  # Number of layers in the DMF
    mlp_num_layer = len(mlp_layers)  # Number of layers in the MLP

    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)
    # Input variables
    user_input = Input(shape=(1, ), dtype='int32', name='user_input')
    item_input = Input(shape=(1, ), dtype='int32', name='item_input')

    # Embedding layer
    user_rating = Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(
        user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(
        item_input)
    user_rating = Reshape((num_items, ))(user_rating)
    item_rating = Reshape((num_users, ))(item_rating)

    # GMF part
    gmf_userlayer = Dense(gmf_num_layer,
                          activation="linear",
                          name='gmf_user_embedding')
    gmf_itemlayer = Dense(gmf_num_layer,
                          activation="linear",
                          name='gmf_item_embedding')
    gmf_user_latent = gmf_userlayer(user_rating)
    gmf_item_latent = gmf_itemlayer(item_rating)

    # Element-wise product of user and item embeddings
    gmf_vector = multiply([gmf_user_latent, gmf_item_latent])

    # DMF part
    dmf_userlayer = Dense(userlayers[0],
                          activation="linear",
                          name='dmf_user_embedding')
    dmf_itemlayer = Dense(userlayers[0],
                          activation="linear",
                          name='dmf_item_embedding')
    dmf_user_latent = dmf_userlayer(user_rating)
    dmf_item_latent = dmf_itemlayer(item_rating)

    #DMF attention
    dmf_attention_user = Dense(userlayers[0],
                               activation='softmax',
                               name='dmf_attention_user')(dmf_user_latent)
    dmf_user_latent2 = multiply([dmf_user_latent, dmf_attention_user],
                                name='dmf_attention_user_mul')
    dmf_user_latent2 = concatenate([dmf_user_latent, dmf_user_latent2])

    dmf_attention_item = Dense(userlayers[0],
                               activation='softmax',
                               name='dmf_attention_item')(dmf_item_latent)
    dmf_item_latent2 = multiply([dmf_item_latent, dmf_attention_item],
                                name='dmf_attention_item_mul')
    dmf_item_latent2 = concatenate([dmf_item_latent, dmf_item_latent2])

    for idx in range(1, dmf_num_layer):
        dmf_userlayer = Dense(userlayers[idx],
                              activation='relu',
                              name='user_layer%d' % idx)
        dmf_itemlayer = Dense(itemlayers[idx],
                              activation='relu',
                              name='item_layer%d' % idx)
        dmf_user_latent2 = dmf_userlayer(dmf_user_latent2)
        dmf_item_latent2 = dmf_itemlayer(dmf_item_latent2)
    dmf_vector = multiply([dmf_user_latent2, dmf_item_latent2])

    # MLP part
    MLP_Embedding_User = Dense(mlp_layers[0] // 2,
                               activation="linear",
                               name='mlp_user_embedding')
    MLP_Embedding_Item = Dense(mlp_layers[0] // 2,
                               activation="linear",
                               name='mlp_item_embedding')
    mlp_user_latent = MLP_Embedding_User(user_rating)
    mlp_item_latent = MLP_Embedding_Item(item_rating)

    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    mlp_attention = Dense(mlp_layers[0],
                          activation='softmax',
                          name='mlp_attention')(mlp_vector)
    mlp_vector2 = multiply([mlp_vector, mlp_attention],
                           name='mlp_attention_mul')
    mlp_vector2 = concatenate([mlp_vector2, mlp_vector])

    for idx in range(1, mlp_num_layer):
        layer = Dense(mlp_layers[idx], activation='relu', name="layer%d" % idx)
        mlp_vector2 = layer(mlp_vector2)

    # Concatenate GMF and DMF and MLP parts
    predict_vector = concatenate([gmf_vector, dmf_vector, mlp_vector2])

    # Final prediction layer
    prediction = Dense(1,
                       activation='sigmoid',
                       kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)

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


def load_pretrain_model(model, gmf_model, dmf_model, dmf_layers, mlp_model,
                        mlp_layers):
    # GMF embeddings
    gmf_user_embeddings = gmf_model.get_layer(
        'gmf_user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer(
        'gmf_item_embedding').get_weights()
    model.get_layer('gmf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('gmf_item_embedding').set_weights(gmf_item_embeddings)

    # DMF embeddings
    dmf_user_embeddings = dmf_model.get_layer(
        'dmf_user_embedding').get_weights()
    dmf_item_embeddings = dmf_model.get_layer(
        'dmf_item_embedding').get_weights()
    model.get_layer('dmf_user_embedding').set_weights(dmf_user_embeddings)
    model.get_layer('dmf_item_embedding').set_weights(dmf_item_embeddings)

    # DMF attention layer
    dmf_attention_users = dmf_model.get_layer('attention_user').get_weights()
    dmf_attention_items = dmf_model.get_layer('attention_item').get_weights()
    model.get_layer('dmf_attention_user').set_weights(dmf_attention_users)
    model.get_layer('dmf_attention_item').set_weights(dmf_attention_items)

    # DMF layers
    for i in range(1, len(dmf_layers)):
        dmf_user_layer_weights = dmf_model.get_layer('user_layer%d' %
                                                     i).get_weights()
        model.get_layer('user_layer%d' % i).set_weights(dmf_user_layer_weights)
        dmf_item_layer_weights = dmf_model.get_layer('item_layer%d' %
                                                     i).get_weights()
        model.get_layer('item_layer%d' % i).set_weights(dmf_item_layer_weights)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer(
        'mlp_user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer(
        'mlp_item_embedding').get_weights()
    model.get_layer('mlp_user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_item_embedding').set_weights(mlp_item_embeddings)

    # MLP attention layer
    mlp_attentions = mlp_model.get_layer('attention_vec').get_weights()
    model.get_layer('mlp_attention').set_weights(mlp_attentions)

    # MLP layers
    for i in range(1, len(mlp_layers)):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    dmf_prediction = dmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate(
        (gmf_prediction[0], dmf_prediction[0][:mlp_layers[-1]],
         mlp_prediction[0]),
        axis=0)
    new_b = gmf_prediction[1] + dmf_prediction[1] + mlp_prediction[1]

    # 1/3 means the contributions of GMF and DMF and MLP are equal
    model.get_layer('prediction').set_weights([new_weights / 3.0, new_b / 3.0])

    return model


if __name__ == '__main__':
    tt = time()
    args = parse_args()
    path = args.path
    dataset = args.dataset
    userlayers = eval(args.userlayers)
    itemlayers = eval(args.itemlayers)
    mlp_layers = eval(args.mlp_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    num_epochs = args.epochs
    verbose = args.verbose
    gmf_pretrain = args.gmf_pretrain
    dmf_pretrain = args.dmf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DeepCF arguments: %s " % args)
    model_out_file = 'Pretrain/%s_ACFNet_%d.h5' % (args.dataset, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(train, num_users, num_items, latent_dim, userlayers,
                      itemlayers, mlp_layers)
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

    # Load pretrain model
    if gmf_pretrain != '' and dmf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(train, num_users, num_items, latent_dim)
        gmf_model.load_weights(gmf_pretrain)

        dmf_model = DMF.get_model(train, num_users, num_items, userlayers,
                                  itemlayers)
        dmf_model.load_weights(dmf_pretrain)

        mlp_model = MLP.get_model(train, num_users, num_items, mlp_layers)
        mlp_model.load_weights(mlp_pretrain)

        model = load_pretrain_model(model, gmf_model, dmf_model, userlayers,
                                    mlp_model, mlp_layers)
        print(
            "Load pretrained GMF (%s) and DMF (%s) and MLP (%s) models done. "
            % (gmf_pretrain, dmf_pretrain, mlp_pretrain))

    # Check Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK,
                                   evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(num_epochs):
        #print("test")
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
        print("The best ACFNet model is saved to %s" % model_out_file)

    print("总时长为")
    print(time() - tt)
