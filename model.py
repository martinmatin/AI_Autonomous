import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
from keras.callbacks import TensorBoard
from time import time
np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #data_dir = dossier 'data'
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    # X contient l'ensemble des noms des images de center, left, right
    X = data_df[['center', 'left', 'right']].values
    # Y contient l'ensemble des valeurs de steering angle
    y = data_df['steering'].values
    print(len(X))
    print('/' * 40)
    # on divise le dataset, test_size = ratio (ici 20%)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    # Le modèle séquentiel est une pile linéaire de couches : A linear stack is a model without any branching. Every layer has one input and output. The output of one layer is the input of the layer below it.
    model = Sequential()
    #Le modèle doit savoir quelle forme d’entrée il doit attendre. C’est la raison pour laquelle la première couche d’un modèle séquentiel doit recevoir les caractéristiques de la forme d’entrée (les autres couches peuvent quant à elles déterminer la forme de leurs entrées par inférence)
    #INPUT_SHAPE = IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
    # Lambda layers in Keras help you to implement layers or functionality that is not prebuilt and which do not require trainable weights
    #the images are normalized (image data divided by 127.5 and subtracted 1.0 pour ramner au range -1 ; 1). As stated in the Model Architecture section, this is to avoid saturation and make gradients work better)
    #keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, weights=None, border_mode='valid', subsample=(1, 1))
 #subsample = avance le filtre de 2 pixel par 2 pixel
 #Pour savoir le nombre la taille de la prochaine fenetre : 1 +  ((taille actuelle - taille filtre) - (taille actuelle - taille filtre)%subsample))/2

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, 5, 5, activation='elu', subsample=(4, 4)))
    model.add(Conv2D(32, 3, 3, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #ModelCheckpoint = Safe the model after each epoch
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
#TensorBoard
    tensorBoard = TensorBoard(log_dir="logs/{}".format(args.test_name))
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid)/args.batch_size,
                        callbacks=[checkpoint,tensorBoard], #fonctions qui sont appelées régulièrement pdt l'entrainement
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    # argparse =  module pour traiter les commandes du shell en orienté objet
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.5)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-e', help='number of epochs',      dest='nb_epoch',          type=int,   default=4)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=2000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='false')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument('-n', help='test_name',             dest='test_name',         type=str,   default=str(time()))
    args = parser.parse_args()


    #Print l'ensemble des paramètres rentrés dans l'objet args
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
