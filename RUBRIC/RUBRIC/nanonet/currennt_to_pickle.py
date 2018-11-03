#!/usr/bin/env python
import argparse
import json
import sys

import numpy as np
from RUBRIC.nanonet import nn
from RUBRIC.nanonet import all_nmers
from RUBRIC.nanonet.cmdargs import FileExist

def get_parser():
    parser = argparse.ArgumentParser(
        description='Convert currennt json network file into pickle. Makes assumptions about meta data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', action=FileExist,
        help='File containing current network')
    parser.add_argument('output', help='Output pickle file')
    
    parser.add_argument("--kmer_length", type=int, default=5,
        help="Length of kmers to learn.")
    parser.add_argument("--bases", type=str, default='ACGT',
        help="Alphabet of kmers to learn.")
    parser.add_argument("--window", type=int, nargs='+', default=[-1, 0, 1],
        help="The detailed list of the entire window.")
    parser.add_argument("--section", type=str, default='template',
        help="Section of read which network is trained against.")
    return parser


def toarray(x):
    return np.ascontiguousarray(np.array(x, order='C', dtype=nn.dtype))


def parse_layer_input(size, weights):
    return None


def parse_layer_feedforward(size, weights, fun):
    M = toarray(weights['input'])
    M = M.reshape((size, -1)).transpose()
    b = toarray(weights['bias'])
    return nn.FeedForward(M, b, fun)


def parse_layer_feedforward_tanh(size, weights):
    return parse_layer_feedforward(size, weights, nn.tanh)


def parse_layer_feedforward_sigmoid(size, weights):
    return parse_layer_feedforward(size, weights, nn.sigmoid)


def parse_layer_feedforward_linear(size, weights):
    return parse_layer_feedforward(size, weights, nn.linear)


def parse_layer_softmax(size, weights):
    M = toarray(weights['input'])
    M = M.reshape((size, -1)).transpose()
    b = toarray(weights['bias'])
    return nn.SoftMax(M, b)


def parse_layer_multiclass(size, weights):
    return None


def parse_layer_blstm(size, weights):
    size = size / 2
    wgts_input = toarray(weights['input']).reshape((4, 2, size, -1)).transpose((0, 1, 3, 2))
    wgts_bias = toarray(weights['bias']).reshape((4, 2, -1))
    wgts_internalMat = toarray(weights['internal'][: 4 * size * size * 2]).reshape((4, 2, size, size)).transpose((0, 1, 3, 2))
    wgts_internalPeep = toarray(weights['internal'][4 * size * size * 2 :]).reshape((3, 2, size))

    iM1 = wgts_input[:, 0, :, :]
    bM1 = wgts_bias[:, 0, :]
    lM1 = wgts_internalMat[:, 0, :, :]
    pM1 = wgts_internalPeep[:, 0, :]
    layer1 = nn.LSTM(iM1, lM1, bM1, pM1)

    iM2 = wgts_input[:, 1, :, :]
    bM2 = wgts_bias[:, 1, :]
    lM2 = wgts_internalMat[:, 1, :, :]
    pM2 = wgts_internalPeep[:, 1, :]
    layer2 = nn.LSTM(iM2, lM2, bM2, pM2)
    return nn.BiRNN(layer1, layer2)


def parse_layer_lstm(size, weights):
    iM = toarray(weights['input']).reshape((4, size, -1)).transpose((0, 2, 1))
    bM = toarray(weights['bias']).reshape((4, size))
    lM = toarray(weights['internal'][ : 4 * size * size]).reshape((4, size, size)).transpose((0, 2, 1))
    pM = toarray(weights['internal'][4 * size * size : ]).reshape((3, size))
    return nn.LSTM(iM, lM, bM, pM)


LAYER_DICT = {'input' : parse_layer_input,
              'blstm' : parse_layer_blstm,
              'feedforward_tanh' : parse_layer_feedforward_tanh,
              'feedforward_logistic' : parse_layer_feedforward_sigmoid,
              'feedforward_identity' : parse_layer_feedforward_linear,
              'lstm' : parse_layer_lstm,
              'blstm' : parse_layer_blstm,
              'softmax' : parse_layer_softmax,
              'multiclass_classification' : parse_layer_multiclass}


def parse_layer(layer_type, size, weights):
    if not layer_type in LAYER_DICT:
        sys.stderr.write('Unsupported layer type {}.\n'.format(layer_type))
        exit(1)
    return LAYER_DICT[layer_type](size, weights)


def network_to_numpy(in_network):
    """Transform a json representation of a network into a numpy
    representation.
    """

    layers = list()
    for layer in in_network['layers']:
        wgts = in_network['weights'][layer['name']] if layer['name'] in in_network['weights'] else None
        layers.append(parse_layer(layer['type'], layer['size'], wgts))
    layers = filter(lambda x: x is not None, layers)

    meta = None
    if 'meta' in in_network:
        meta = in_network['meta']
    network = nn.Serial(layers)
    network.meta = meta
    return network


if __name__ == '__main__':
    args = get_parser().parse_args() 

    try:
        with open(args.input, 'r') as fh:
            in_network = json.load(fh)
    except:
        sys.stderr.write('Failed to read from {}.\n'.format(args.input))
        exit(1)

    if not 'layers' in in_network:
        sys.stderr.write('Could not find any layers in {} -- is it a network file?\n'.format(args.network))
        exit(1)
    if not 'weights' in in_network:
        sys.stderr.write('Could not find any weights in {} -- is network trained?\n'.format(args.network))
        exit(1)

    # Build meta, taking some guesses
    kmers = all_nmers(args.kmer_length, alpha=args.bases)
    kmers.append('X'*args.kmer_length)
    in_network['meta'] = {
        'window':args.window,
        'n_features':in_network['layers'][0]['size'],
        'kmers':kmers,
        'section':args.section
    }

    network = network_to_numpy(in_network)
    np.save(args.output, network)
