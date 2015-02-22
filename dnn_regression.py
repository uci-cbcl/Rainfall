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
import pylearn2

def initialize_dnn(activation_function, num_features):
    if activation_function=='relu':
        self.hidden_layer = pylearn2.models.mlp.RectifiedLinear
    elif activation_function=='tanh':
        self.hidden_layer = pylearn2.models.mlp.Tanh
    elif activation_function=='sig':
        self.hidden_layer = pylearn2.models.mlp.Sigmoid

    layers = list()
    for l in xrange(self.num_hidden_layers):
        layers.append(self.hidden_layer(layer_name='h'+str(l), dim=self.num_hidden_nodes_per_layer,
        sparse_init=15))
    layers.append(pylearn2.models.mlp.Linear(layer_name='y', dim=1, irange=0.))
    
    model = pylearn2.models.mlp.MLP(layers=layers, nvis=num_features)
    
    cost = pylearn2.costs.cost.SumOfCosts(costs=[pylearn2.costs.mlp.Default(),
        pylearn2.costs.mlp.WeightDecay(coeffs=[.00005]*self.num_layers)])

    criteria = [pylearn2.termination_criteria.MonitorBased(channel_name="valid_objective",
        prop_decrease=0., N=10),
        pylearn2.termination_criteria.EpochCounter(max_epochs=100)]
    
    algorithm = pylearn2.training_algorithms.sgd.SGD(batch_size=100, learning_rate=.01,
        monitoring_dataset={'train':dataset_tr, 'valid':dataset_va, 'test':dataset_te},
        cost=cost,
        learning_rule=pylearn2.training_algorithms.learning_rule.Momentum(
        init_momentum=.5),
        termination_criterion=pylearn2.termination_criteria.And(
        criteria=criteria))

    extensions = [pylearn2.train_extensions.best_params.MonitorBasedSaveBest(
        channel_name='valid_objective',
        save_path=self.output_dir+ "NNModel_best_" + str(iteration) + ".pkl"),
        pylearn2.training_algorithms.learning_rule.MomentumAdjustor(start=1, saturate=10,
        final_momentum=.99)]

    trainer = pylearn2.train.Train(dataset=dataset_tr, model=self.pl2model,
        algorithm=algorithm, extensions=extensions,
        save_path=output_dir+ "NNModel_" + str(iteration) + ".pkl", save_freq=1)
    return trainer

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

def train(input_dir, output_dir, useX1andX2):
    Xtr, Ytr, Xte = load_data(input_dir, useX1andX2)
    s = scale_data(Xtr, Xte)
    model = initialize_dnn()

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
    default='tanh',
    help='The activation function to use. relu, tanh, sig (default: tanh).')
    
    parser.add_argument('--numhiddenlayers', '-l', type=int, default=1,
    help='The number of hidden layers (default: 1).')
    
    parser.add_argument('--numhiddennodesperlayer', '-k', type=int, default=100,
    help='The number of hidden nodes per layer (default: 100).')

    parser.add_argument('--useX1andX2', '-u', action='store_true',
    help='If specified, use both X1 and X2 features. Otherwise, use only X1.')     

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

