#!/usr/bin/env python

import argparse
import json
import os
import sys
import pkg_resources
import tempfile
import numpy as np

from RUBRIC.nanonet import run_currennt_noisy
from RUBRIC.nanonet.cmdargs import FileExist, AutoBool
from RUBRIC.nanonet import iterate_fast5
from RUBRIC.nanonet import make_currennt_training_input_multi
from RUBRIC.nanonet import random_string, conf_line, tang_imap
from RUBRIC.nanonet import network_to_numpy


def get_parser():
    parser = argparse.ArgumentParser(
        description="A simple ANN training wrapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--train", action=FileExist,
        help="Input training data, either a path to fast5 files or a single netcdf file", required=True)
    parser.add_argument("--train_list", action=FileExist, default=None,
        help="Strand list constaining training set")
    parser.add_argument("--section", default='template', choices=('template', 'complement'),
        help="Section of reads to train")
    
    parser.add_argument("--val", action=FileExist,
        help="Input validation data, either a path to fast5 files or a single netcdf file", required=True)
    parser.add_argument("--val_list", action=FileExist, default=None,
        help="Strand list constaining validation set")
    parser.add_argument("--workspace", default=tempfile.gettempdir(),
        help="Path for storing training and validation NetCDF files, if not specified a temporary file is used.")
    
    parser.add_argument("--output", help="Output prefix", required=True)

    parser.add_argument("--model", action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_model.tmpl'),
        help="ANN configuration file")
    parser.add_argument("--kmer_length", type=int, default=5,
        help="Length of kmers to learn.")
    parser.add_argument("--bases", type=str, default='ACGT',
        help="Alphabet of kmers to learn.")

    parser.add_argument("--device", type=int, default=0,
        help="ID of CUDA device to use.")
    parser.add_argument("--cuda", default=False, action=AutoBool,
        help="Use CUDA acceleration.")
    parser.add_argument("--window", type=int, nargs='+', default=[-1, 0, 1],
        help="The detailed list of the entire window.")
    
    training_parameter_group = parser.add_argument_group("Training Parameters.")
    training_parameter_group.add_argument("--max_epochs", type=int, default=500,
        help="Max training epocs, default 500")
    training_parameter_group.add_argument("--max_epochs_no_best", type=int, default=50,
        help="Stop training when no improvment for number of epocs, default 50" )
    training_parameter_group.add_argument("--validate_every", type=int, default=5,
        help="Run validation data set every number of epocs.")
    training_parameter_group.add_argument("--parallel_sequences", type=int, default=125,
        help="Number of sequences in a min-batch")
    training_parameter_group.add_argument("--learning_rate", type=float, default=1e-5,
        help="Learning rate parameters of SGD." )
    training_parameter_group.add_argument("--momentum", type=float, default=0.9,
        help="Momentum parameter of SGD." )
    training_parameter_group.add_argument("--cache_path", default=tempfile.gettempdir(),
        help="Path for currennt temporary files.")

    return parser


def prepare_input_file(in_out, **kwargs):
    path, in_list, output = in_out 

    print "Creating training data NetCDF: {}".format(output)
    fast5_files = list(iterate_fast5(path, paths=True, strand_list=in_list))
    return make_currennt_training_input_multi(
        fast5_files=fast5_files, 
        netcdf_file=output,
        **kwargs
    )


def main():
    if len(sys.argv) == 1: 
        sys.argv.append("-h")
    args = get_parser().parse_args()

    if not args.cuda:
        args.nseqs = 1
   
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)
 
    # file names for training
    tag = random_string()
    modelfile  = os.path.abspath(args.model)
    outputfile = os.path.abspath(args.output)
    temp_name = os.path.abspath(os.path.join(
        args.workspace, 'nn_data_{}_'.format(tag)
    ))
    config_name = os.path.abspath(os.path.join(
        args.workspace, 'nn_{}.cfg'.format(tag)
    ))
    
    # Create currennt training input files
    trainfile = '{}{}'.format(temp_name, 'train.netcdf') 
    valfile = '{}{}'.format(temp_name, 'validation.netcdf')
    inputs = (
        (args.train, args.train_list, trainfile),
        (args.val, args.val_list, valfile),
    )
    fix_kwargs = {
        'window':args.window,
        'kmer_len':args.kmer_length,
        'alphabet':args.bases,
        'callback_kwargs':{'section':args.section, 'kmer_len':args.kmer_length}
    }
    for results in tang_imap(prepare_input_file, inputs, fix_kwargs=fix_kwargs, threads=2):
        n_chunks, n_features, out_kmers = results
        if n_chunks == 0:
            raise RuntimeError("No training data written.")


    # fill-in templated items in model
    n_states = len(out_kmers)
    with open(modelfile, 'r') as model:
        mod = model.read()
    mod = mod.replace('<section>', args.section)
    mod = mod.replace('<n_features>', str(n_features))
    mod = mod.replace('<n_states>', str(n_states))
    try:
        mod_meta = json.loads(mod)['meta']
    except Exception as e:
        mod_meta = dict()
    mod_meta['n_features'] = n_features
    mod_meta['kmers'] = out_kmers
    mod_meta['window'] = args.window

    modelfile = os.path.abspath(os.path.join(
        args.workspace, 'input_model.jsn'
    ))
    with open(modelfile, 'w') as model:
        model.write(mod)
    final_network = "{}_final.jsn".format(outputfile)
    best_network_prefix = "{}_auto".format(outputfile)
    # currennt appends some bits here

    # currennt cfg files
    with open(config_name, 'w') as currennt_cfg:
        if not args.cuda:
            currennt_cfg.write(conf_line('cuda', 'false'))
        # IO
        currennt_cfg.write(conf_line("cache_path", args.cache_path))
        currennt_cfg.write(conf_line("network", modelfile))
        currennt_cfg.write(conf_line("train_file", trainfile))
        currennt_cfg.write(conf_line("val_file", valfile))
        currennt_cfg.write(conf_line("save_network", final_network))
        currennt_cfg.write(conf_line("autosave_prefix", best_network_prefix))
        # Tunable parameters
        currennt_cfg.write(conf_line("max_epochs", args.max_epochs))
        currennt_cfg.write(conf_line("max_epochs_no_best", args.max_epochs_no_best))
        currennt_cfg.write(conf_line("validate_every", args.validate_every))
        currennt_cfg.write(conf_line("parallel_sequences", args.parallel_sequences))
        currennt_cfg.write(conf_line("learning_rate", args.learning_rate))
        currennt_cfg.write(conf_line("momentum", args.momentum))
        # Fixed parameters
        currennt_cfg.write(conf_line("train", "true"))
        currennt_cfg.write(conf_line("weights_dist", "normal"))
        currennt_cfg.write(conf_line("weights_normal_sigma", "0.1"))
        currennt_cfg.write(conf_line("weights_normal_mean", "0"))
        currennt_cfg.write(conf_line("stochastic", "true"))
        currennt_cfg.write(conf_line("input_noise_sigma", "0.0"))
        currennt_cfg.write(conf_line("shuffle_fractions", "false"))
        currennt_cfg.write(conf_line("shuffle_sequences", "true"))
        currennt_cfg.write(conf_line("autosave_best", "true"))
    
    # run currennt
    print "\n\nRunning currennt with: {}".format(config_name)
    run_currennt_noisy(config_name, device=args.device)

    # Currennt won't pass through our meta in the model, amend the output
    # and write out a numpy version of the network
    best_network = "{}.best.jsn".format(best_network_prefix)
    best_network_numpy = "{}_best.npy".format(outputfile)

    print "Adding model meta to currennt best network: {}".format(best_network)
    mod = json.load(open(best_network, 'r'))
    mod['meta'] = mod_meta
    json.dump(mod, open(best_network, 'w'))
    print "Transforming network to numpy pickle: {}".format(best_network_numpy)
    mod = network_to_numpy(mod)
    np.save(best_network_numpy, mod)
        


if __name__ == '__main__':
    main() 
