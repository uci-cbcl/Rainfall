#!/usr/bin/env python

"""
Script for deep neural network regression to predict rainfall amount.

CS 273A
In-class Kaggle
"""

# Standard library imports
import argparse
import errno
import os
import sys

# Third party imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import pylearn2
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def predict(x, model):
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    f = function([X], Y)
    return f(x)

def initialize_dnn(dataset_tr, dataset_va, output_dir, activation_function, num_features, 
    num_hidden_layers, num_hidden_nodes_per_layer, learning_rate, minibatch_size, 
    stdev, dropout, gaussian):

    output_dir = output_dir + "/"

    if activation_function=='relu':
        hidden_layer = pylearn2.models.mlp.RectifiedLinear
    elif activation_function=='tanh':
        hidden_layer = pylearn2.models.mlp.Tanh
    elif activation_function=='sig':
        hidden_layer = pylearn2.models.mlp.Sigmoid

    layers = list()
    for l in xrange(num_hidden_layers):
        layers.append(hidden_layer(layer_name='h'+str(l), dim=num_hidden_nodes_per_layer,
        istdev=stdev))
    layers.append(pylearn2.models.mlp.Linear(layer_name='y', dim=1, istdev=stdev))
    
    model = pylearn2.models.mlp.MLP(layers=layers, nvis=num_features)
    
    costFunction = pylearn2.costs.mlp.Default()
    if dropout:
        costFunction = pylearn2.costs.mlp.dropout.Dropout()
    cost = pylearn2.costs.cost.SumOfCosts(costs=[costFunction])

    criteria = [pylearn2.termination_criteria.MonitorBased(channel_name="train_objective",
        prop_decrease=0., N=10),
        pylearn2.termination_criteria.EpochCounter(max_epochs=10000)]
    
    algorithm = pylearn2.training_algorithms.sgd.SGD(batch_size=minibatch_size, 
        learning_rate=learning_rate,
        monitoring_dataset={'train':dataset_tr, 'valid':dataset_va},
        cost=cost,
        termination_criterion=pylearn2.termination_criteria.And(
        criteria=criteria))

    extensions = [pylearn2.train_extensions.best_params.MonitorBasedSaveBest(
        channel_name='train_objective',
        save_path=output_dir+ "NNModel_best.pkl")]

    trainer = pylearn2.train.Train(dataset=dataset_tr, model=model,
        algorithm=algorithm, extensions=extensions,
        save_path=output_dir+ "NNModel.pkl", save_freq=1)

    return trainer, model

def scale_data(Xtr, Xte):
    s = StandardScaler()
    s.fit(Xtr)
    np.copyto(Xtr,s.transform(Xtr))
    np.copyto(Xte,s.transform(Xte))
    return s

def load_data(input_dir, useX1andX2):
    input_dir = indput_dir + "/"
    Xtr = numpy.loadtxt('kaggle.X1.train.txt',delimiter=',')    
    Xte = numpy.loadtxt('kaggle.X1.test.txt',delimiter=',')
    Ytr = numpy.loadtxt('kaggle.Y.train.txt')    
    if useX1andX2:
        X2tr = numpy.loadtxt('kaggle.X2.train.txt',delimiter=',')
        X2te = numpy.loadtxt('kaggle.X2.test.txt',delimiter=',')
        Xtr = np.concatenate((Xtr, X2tr), axis=1)
        Xte = np.concatenate((Xte, X2te), axis=1)    
    return Xtr, Ytr, Xte

def train(input_dir, output_dir, activation_function, num_hidden_layers, 
    num_hidden_nodes_per_layer, learning_rate, minibatch_size, stdev, 
    dropout, useX1andX2, gaussian, valid_size, random_seed):
    
    np.random.seed(random_seed)

    Xtr, Ytr, Xte = load_data(input_dir, useX1andX2)
    s = scale_data(Xtr, Xte)

    Xtr, Xva, Ytr, Yva = train_test_split(Xtr, Ytr, test_size=valid_size)

    dataset_tr = DenseDesignMatrix(X=Xtr,y=Ytr)
    dataset_va = DenseDesignMatrix(X=Xva,y=Yva)

    _, num_features = Xtr.shape

    trainer, model = initialize_dnn(dataset_tr, dataset_va, output_dir, 
        activation_function, num_features, num_hidden_layers, 
        num_hidden_nodes_per_layer, learning_rate, minibatch_size,
        stdev, dropout, gaussian)
    trainer.main_loop()

    Ytr_pred = predict(Xtr, model)
    Yva_pred = predict(Xva, model)
    Yte_pred = predict(Xte, model)

    test_predictions_file = open(output_dir + "/predictions.csv", "w")
    test_predictions_file.write('ID,Prediction\n')
    for i in xrange(length(Yte_pred)):
        test_predictions_file.write(str(i+1) + "," + str(Yte_pred[i]) + "\n")
    test_predictions_file.close()

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
    description="Train a deep network model for genome segmentation from a directory of input "
    "files, such as binarized ChromHMM training.",
    epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
    formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--inputdir', '-i', type=str, required=True,
    help='Directory containing input Kaggle training and test files.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-o", "--outputdir", type=str,
    help='The output directory. Will cause an error if the directory already exists.')
    group.add_argument("-oc", "--outputdirc", type=str,
    help='The output directory. Will overwrite if directory already exists.')
    
    parser.add_argument('--activation', '-a', action='store', choices=("tanh","relu","logistic"),
    default='relu',
    help='The activation function to use. relu, tanh, sig (default: relu).')
    
    parser.add_argument('--numhiddenlayers', '-l', type=int, default=1,
    help='The number of hidden layers (default: 1).')
    
    parser.add_argument('--numhiddennodesperlayer', '-k', type=int, default=100,
    help='The number of hidden nodes per layer (default: 100).')

    parser.add_argument('--minibatchsize', '-m', type=int, default=100,
    help='Minibatch size (default: 100).')

    parser.add_argument('--useX1andX2', '-u', action='store_true',
    help='If specified, use both X1 and X2 features. Otherwise, use only X1.')     

    parser.add_argument('--dropout', '-d', action='store_true',
    help='If specified, use dropout.')

    parser.add_argument('--learningrate', '-l', type=float, default=0.01,
    help='Learning rate (default: 0.01).')

    parser.add_argument('--randomseed', '-r', type=int, default=0,
    help='Random seed (default: 0).')

    parser.add_argument('--stddev', '-s', type=float, default=0.01,
    help='Random seed (default: 0.01).')

    parser.add_argument('--validsize', '-v', type=float, default=0.1,
    help='Fraction of training set to set aside as validation set. (default: 0.1).')

    parser.add_argument('--gaussian', '-g', action='store_true',
    help='If specified, use linear gaussian output layer.')

    return parser

if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.outputdir is None:
        clobber = True
        output_dir = args.outputdirc
    else:
        clobber = False
        output_dir = args.outputdir

    try:#adapted from DREME.py by T. Bailey
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            if not clobber:
                print >> sys.stderr, ("output directory (%s) already exists "
                "but program was told not to clobber it") % (output_dir);
                sys.exit(1)
            else:
                print >> sys.stderr, ("output directory (%s) already exists "
                "so it will be clobbered") % (output_dir);

    train(args.inputdir, args.outputdir, args.activationfunction, args.numhiddenlayers, 
        args.numhiddennodesperlayer, args.learningrate, args.minibatchsize, args.stdev, 
        args.dropout, args.useX1andX2, args.gaussian, args.validsize, args.randomseed)
