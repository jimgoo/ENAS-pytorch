import os
import sys
import ipdb
import time
import math
import json
import argparse
import contextlib
import pandas as pd
import numpy as np
from itertools import count
from collections import OrderedDict
import dill as pickle
import shutil

import torch as to
import torch.autograd as ag
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
mp.set_start_method('spawn')
pickle.settings['byref'] = True

from amb.settings import get_logger, settings
# from amb.attachments.genome.activation import activations as torch_activations
from amb.data.profiler import DataProfiler
from amb.attachments.genome.deep.genes import LinearLayer, LSTM
from amb.attachments.genome.deep.genome import DeepGenome, DeepGenomeConfig
from amb.torchnet.deepnet import DeepNet
from amb.model.supervised.traits import ModelTraits
from amb.utils import LossFuncs
from amb.plot import graph_deep_net
from amb.model.supervised.callbacks import TensorBoardCallback

# these are in ENAS-torch
import utils

logger = get_logger(__name__)

parser = argparse.ArgumentParser(description='NAS')

parser.add_argument('--model', type=str, default='nas', help='nas | amb | both')
parser.add_argument('--dataset', type=str, default='all')  # default='Datasets.UCI.UCI_beijing')
parser.add_argument('--model_dir', type=str, default='logs')

parser.add_argument('--controller_hid', type=int, default=100)

parser.add_argument('--controller_grad_clip', type=float, default=0)

parser.add_argument('--controller_max_step', type=int, default=2000,
                    help='step for controller parameters (per epoch)')

parser.add_argument('--log_step', type=int, default=50,
                    help='How often to reset the controllers history')

parser.add_argument('--log_step_genome', type=int, default=20,
                    help='how often to log the best genome (default to 20 for pop_size in AMB)')

parser.add_argument('--nprocs', type=int, default=8,
                    help='Number of preprocessors to use to evaluate genomes')

parser.add_argument('--genome_epochs', type=int, default=200,
                    help='max epochs per genome')

parser.add_argument('--max_time', type=int, default=None,
                    help='max time to run in seconds')

parser.add_argument('--max_generations', type=int, default=10,
                    help='max generations')

parser.add_argument('--controller_path', type=str, default='',
                    help='path to pretrained controller')

# not sure about these
# parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
# parser.add_argument('--entropy_coeff', type=float, default=1e-4)
# parser.add_argument('--softmax_temperature', type=float, default=5.0)

args = parser.parse_args()
args.cuda = True


class Controller(to.nn.Module):

    def __init__(self, n_inputs, n_outputs, neatcfg, vocab, activations):
        super(Controller, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neatcfg = neatcfg
        self.vocab = vocab
        self.activations = activations

        self.num_tokens = [len(self.vocab), len(self.activations), 2]
        self.num_total_tokens = sum(self.num_tokens)

        self.encoder = to.nn.Embedding(self.num_total_tokens, args.controller_hid)

        self.lstm = to.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # RNN will alternate between picking an activation and a vocab member,
        # so each decoder gets used on alternating forward passes.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = to.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)
        self._decoders = to.nn.ModuleList(self.decoders)

        self.reset_parameters()

        # doesn't include the final output layer
        self.set_max_layers(2)

    def set_max_layers(self, max_layers):
        self.max_layers = max_layers  # not including Linear ouput layer
        self.num_blocks = len(self.num_tokens) * self.max_layers

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        zeros = to.zeros(batch_size, args.controller_hid)
        return (utils.get_variable(zeros, args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), args.cuda, requires_grad=False))

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)
        # logits /= args.softmax_temperature ## TODO: what effect does this have?
        return logits, (hx, cx)

    def sample(self, batch_size=1):
        if batch_size != 1:
            raise Exception('Haven''t tested batch_size > 1 yet')

        # print('*' * 100)
        # print('Sampling')
        # print('num_tokens', self.num_tokens)

        # [B, L, H]
        # inputs = self.static_inputs[batch_size]
        # hidden = self.static_init_hidden[batch_size]
        inputs = utils.get_variable(to.zeros(batch_size, args.controller_hid),
                                    args.cuda, requires_grad=False)
        hidden = self.init_hidden(batch_size)

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []

        n_decoders = len(self.num_tokens)

        # for block_idx in range(2*(self.args.num_blocks - 1) + 1):
        # for block_idx in range(n_decoders, self.num_blocks + n_decoders)

        for block_idx in range(self.num_blocks):

            # 0: function/activation, 1: previous node/layer
            mode = block_idx % n_decoders

            # print('block_idx', block_idx, 'mode', mode)

            logits, hidden = self.forward(inputs,
                                          hidden,
                                          mode,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data

            # select the log_prob for the action taken
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

            """
            ## above is the same as:
            from torch.distributions import Categorical
            m = Categorical(probs)
            action = m.sample()
            selected_log_prob = m.log_prob(action)
            """

            assert selected_log_prob.numel() == 1
            selected_log_prob = selected_log_prob.view(-1)

            assert action.numel() == 1
            action = action.view(-1)

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

            # Normally for RNNs in ENAS, `self.num_tokens` alternates bewteeen `n_activations` and `n_other_blocks`
            # self.num_tokens = [4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4]

            inputs = utils.get_variable(
                action + sum(self.num_tokens[:mode]),
                requires_grad=False)

            # print('block_idx', block_idx, 'mode', mode,
            #       'action', action.cpu().numpy(),
            #       'inputs', inputs.cpu().data.numpy()[0])

            # end token
            if inputs.item() == self.num_total_tokens - 1:
                break

            if mode == 0:
                prev_nodes.append(action)
            elif mode == 1:
                activations.append(action)
            elif mode == 2:
                # mode to check for stopping
                pass

        # transposes just flattens things out (cat does same thing)
        # prev_nodes = to.stack(prev_nodes).transpose(0, 1)
        # activations = to.stack(activations).transpose(0, 1)

        prev_nodes = to.cat(prev_nodes)
        activations = to.cat(activations)
        log_probs = to.cat(log_probs)
        entropies = to.cat(entropies)

        genome = self._construct_dags(prev_nodes, activations)

        ret_vals = (genome, log_probs, entropies)

        return ret_vals

    def _construct_dags(self, prev_nodes, activations):

        prev_nodes = list(map(int, prev_nodes.cpu().numpy().reshape((-1,))))
        activations = list(map(int, activations.cpu().numpy().reshape((-1,))))

        # print('-' * 100)
        # print('prev_nodes', prev_nodes)
        # print('activations', activations)

        if len(prev_nodes) != len(activations):
            raise Exception('Differing lengths: prev_nodes: {}, activations: {}'.format(
                len(prev_nodes), len(activations)))

        # end = min(len(prev_nodes), len(activations))
        # prev_nodes = prev_nodes[:end]
        # activations = activations[:end]

        # print('prev_nodes2', prev_nodes)
        # print('activations2', activations)

        # Create a DeepGenome's nodes and connections from the sampled DAG
        genome = DeepGenome(1, self.neatcfg)

        nodes = OrderedDict()
        connections = OrderedDict()

        # create input gene
        input_gene = LinearLayer(key=-1)
        input_gene.key_orig = input_gene.key
        input_gene.numunits = self.n_inputs
        input_gene.enabled = True
        input_gene.activation = 'identity'
        nodes[-1] = input_gene
        connections[(-1, 1)] = input_gene

        # create intermediate nodes and connections
        key = 1
        for i, (gene_idx, act_idx) in enumerate(zip(prev_nodes, activations)):

            gene = list(self.vocab.values())[gene_idx]
            gene.activation = self.activations[act_idx]

            gene.key_orig = gene.key
            gene.key = key
            nodes[key] = gene

            if i != len(prev_nodes) - 1:
                # not the last node
                conn = (key, key + 1)
                key += 1
            else:
                # last node, connect to output gene's key
                conn = (key, 0)
            connections[conn] = gene

            # print(i, gene.key, gene, gene.activation, conn)

        # create output gene
        output_gene = LinearLayer(key=0)
        output_gene.key_orig = output_gene.key
        output_gene.numunits = self.n_outputs
        output_gene.enabled = True
        output_gene.activation = 'identity'
        nodes[0] = output_gene

        genome.nodes = nodes
        genome.connections = connections

        return genome

    def save(self):
        path = os.path.join(args.model_dir, 'controller.to')
        to.save(self.state_dict(), path)
        logger.info('[*] SAVED controller to: {}'.format(path))


class SimpleNAS:

    def __init__(self, profile=None):
        self.profile = profile
        self.max_epochs = 100

        self.is_class = self.profile.has_categorical_target()
        self.recurrent = self.profile.is_timeseries

        self.controller_step = 0
        self.best_genome, self.best_model = None, None
        self.n_models = 0

        self.history = []
        self.history_file = os.path.join(args.model_dir, 'history.json')

    def set_model_traits(self):

        if self.is_class:
            # Default for classification
            loss_func = LossFuncs.CrossEntropy.name
        else:
            # Default for regression
            loss_func = LossFuncs.MSE.name

        loss_function = LossFuncs[loss_func]

        tp = self.profile.get_target_profile()
        class_cols = tp['is_cat'].values.astype(bool)

        traits = ModelTraits(recurrent=self.recurrent,
                             loss_function=loss_function,
                             fitness_fn_name=None,
                             is_class=self.is_class,
                             class_cols=class_cols,
                             reset=0,
                             neatcfg=self.neatcfg,
                             leea_decay=None,
                             sample_prop=None,
                             preprocessor=None,
                             val_size=0.3)
        self.traits = traits

    def build_space(self):

        activations = ['sigmoid', 'tanh', 'gauss', 'relu', 'identity']

        layers = ['linear']

        attention = False
        if self.recurrent:
            layers += ['lstm']
            seqlens = [2**i for i in range(3, 11) if 2**i <= self.traits.batch_size]

        sizes = [2**i for i in range(1, 11)]

        logger.debug('search space params: ' + '-' * 20)
        logger.debug('activations:    %s', activations)
        logger.debug('sizes:          %s', sizes)
        logger.debug('layers:         %s', layers)
        if self.recurrent:
            logger.debug('lstm_attention: %s', attention)
            logger.debug('lstm_seqlens:   %s', seqlens)

        vocab = OrderedDict()
        for size in sizes:
            gene = LinearLayer(len(vocab) + 1)
            gene.numunits = size
            gene.enabled = True
            vocab[gene.key] = gene

            if self.recurrent:
                for n_layers in range(1, 3):
                    for seqlen in seqlens:
                        gene = LSTM(len(vocab) + 1)
                        gene.numunits = size  # TODO: create a different set ?
                        gene.numlayers = n_layers
                        gene.attention = attention
                        gene.seqlength = seqlen
                        gene.enabled = True
                        vocab[gene.key] = gene

        self.vocab = vocab
        self.activations = activations

        vocab_path = os.path.join(args.model_dir, 'vocab.pkl')
        logger.info('Writing vocab to: {}'.format(vocab_path))
        with open(vocab_path, 'wb') as fout:
            fout.write(pickle.dumps({'vocab': self.vocab, 'activations': self.activations}))

    def fit(self, X, y, use_tensorboard=True):

        n_inputs = X.shape[1]
        n_outputs = self.profile.num_target_classes()
        if n_outputs == 0:
            n_outputs = 1

        self.neatcfg = utils.Config(n_inputs)
        self.amb = utils.AMB(self.neatcfg)  # for plotting deep networks
        self.set_model_traits()
        self.build_space()

        if use_tensorboard:
            self.tb = TensorBoardCallback(args.model_dir, self.amb)
        else:
            self.tb = None

        # self.traits.batch_size = 256

        idx_train, idx_val = self.traits.get_cv_split(X.values, y.values)
        idx_train, idx_val = to.LongTensor(idx_train), to.LongTensor(idx_val)

        logger.debug('train size: {}, val size: {}'.format(len(idx_train), len(idx_val)))

        X = X.values
        y = y.values
        X = to.Tensor(X.astype('float32'))
        if self.is_class:
            y = y.dot(np.arange(y.shape[1])).astype(int)
            y = to.LongTensor(y)
        else:
            y = to.Tensor(y.astype('float32'))

        X = ag.Variable(X)
        y = ag.Variable(y)

        self.x = X[idx_train]
        self.y = y[idx_train]
        self.x_val = X[idx_val]
        self.y_val = y[idx_val]

        self.controller = Controller(n_inputs, n_outputs, self.neatcfg, self.vocab, self.activations)
        self.controller.cuda()

        if args.controller_path != '':
            logger.info('Loading controller weights from: {}'.format(args.controller_path))
            self.controller.load_state_dict(to.load(args.controller_path))

        # print controller and save string rep to file
        print(self.controller)
        num_params = utils.num_params(self.controller)
        num_params = 'num_params: {:,}'.format(num_params)
        print(num_params)
        path = os.path.join(args.model_dir, "controller.txt")
        with open(path, 'w') as fp:
            fp.write(str(self.controller) + '\n' + num_params)

        # TODO: try higher
        # self.controller_lr = 3.5e-4  # the default
        self.controller_lr = 1e-3
        self.controller_optim  = to.optim.Adam(self.controller.parameters(), lr=self.controller_lr)

        if args.nprocs > 1:
            self.pool = mp.Pool(args.nprocs)

        self.start_time = time.time()
        self.epoch = 1

        stop = self.train_controller()
        logger.info('model_dir: %s', args.model_dir)

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """

        self.controller.train()
        # TODO(brendan): Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []
        genomes_all = []

        # hidden = self.shared.init_hidden(self.args.batch_size)
        total_loss = 0
        genome_epochs = args.genome_epochs

        for step in range(args.controller_max_step):

            bp_args = []
            log_probs_all = []

            for i in range(args.nprocs):
                # sample models
                genome, log_probs, entropies = self.controller.sample()

                # keep track of all log_probs for each genome
                log_probs_all.append(log_probs)

                # calculate reward
                np_entropies = entropies.data.cpu().numpy()

                # append entropies for all models
                entropy_history.extend(np_entropies)

                # rewards, valid_loss, genome_model = utils.get_reward(genome, np_entropies, self.traits, self.x, self.y, self.x_val, self.y_val)
                bp_args.append((genome, np_entropies, self.traits, self.x,
                                self.y, self.x_val, self.y_val, genome_epochs))

            if args.nprocs == 1:
                rewards_batch = [utils.get_reward(*bp_args[0])]
            else:
                rewards_batch = self.pool.starmap(utils.get_reward, bp_args)

            for i, (rewards, valid_loss, genome, model, bp_iters) in enumerate(rewards_batch):

                genomes_all.append((genome, float(valid_loss), self.controller_step, bp_iters))

                # check for the best model
                self.n_models += 1
                if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                    self.best_genome = genome
                    self.best_model = model

                # print('-'*100)
                # print('genome.trained', genome.trained)
                # for module in model.modules:
                #     id_ = module.id_
                #     print('id_', id_)
                #     print(genome.nodes[id_].parameters.keys())

                # print('*'*100)
                # for key, gene in genome.nodes.items():
                #     gene_orig = self.vocab.get(gene.key_orig, None)
                #     print(key, gene.key_orig, gene_orig, gene)
                #     if gene_orig is not None:
                #         #function = to.nn.Linear(input_units, self.numunits)
                #         ipdb.set_trace()
                #     #    self.vocab[node.key_orig].set_parameters(gene.function)
                # print('*'*100)

                # print('*'*100)
                # for key, module in model.genome.nodes.items():
                #     key_orig = module.key_orig
                #     gene_orig = self.vocab.get(key_orig, None)
                #     if gene_orig is not None:
                #         module = module.torch()
                #         self.vocab[key_orig].set_parameters(module)
                # print('*'*100)

                """
                hist = {'time': time.time(),
                        'loss': float(valid_loss),
                        'model': str(model),
                        'num_params': utils.num_params(model)}
                self.history.append(hist)
                
                # wirte history to JSON for further analysis
                with open(self.history_file, 'a') as fout:
                    fout.write(json.dumps(hist) + '\n')
                """

                # VIP: you get one reward per entropy
                # reward_history.extend(rewards)

                """
                I have to do mean of entropies here because the number of 
                entropies varies based on network depth:
                    R = 10 - valid_loss
                    rewards = R + 1e-4 * entropies
                """
                rewards = np.mean(rewards)
                reward_history.append(rewards)

                # moving average baseline
                if baseline is None:
                    baseline = rewards
                else:
                    decay = 0.95  # ema_baseline_decay (very important)
                    baseline = decay * baseline + (1 - decay) * rewards

                adv = rewards - baseline
                # adv_history.extend(adv)
                adv_history.append(adv)

                # policy loss
                # loss = -log_probs_all[i] * utils.get_variable(adv, args.cuda, requires_grad=False)
                loss = -log_probs_all[i] * utils.get_variable([adv], args.cuda, requires_grad=False)

                # if args.entropy_mode == 'regularizer':
                #     loss -= args.entropy_coeff * entropies

                loss = loss.sum()  # or loss.mean()

                # update
                self.controller_optim.zero_grad()
                loss.backward()

                if args.controller_grad_clip > 0:
                    to.nn.utils.clip_grad_norm(self.controller.model.parameters(), args.controller_grad_clip)

                self.controller_optim.step()
                self.controller_step += 1
                assert self.controller_step == self.n_models, (self.controller_step, self.n_models)

                total_loss += utils.to_item(loss.data)

                # if ((step % args.log_step) == 0) and (step > 0):
                # if ((step * args.nprocs % args.log_step) == 0) and (step > 0):
                if self.controller_step % args.log_step == 0:

                    logger.info('-' * 100)
                    logger.info('summarizing and resetting: reward_history, adv_history, entropy_history')
                    logger.info('step: {}, controller_step: {}, n_models: {}, max_layers: {}'.format(
                        step, self.controller_step, self.n_models, self.controller.max_layers))
                    logger.info('len(reward_history): {}, len(adv_history): {}, len(entropy_history): {}, len(genomes_all): {}'.format(
                        len(reward_history), len(adv_history), len(entropy_history), len(genomes_all)))

                    self._summarize_controller_train(total_loss,
                                                     adv_history,
                                                     entropy_history,
                                                     reward_history,
                                                     avg_reward_base)

                    reward_history, adv_history, entropy_history = [], [], []
                    total_loss = 0

                    # update max number of layers (do it here so stats are comparable)
                    self.controller.set_max_layers(self.controller.max_layers + 1)

                    self.controller.save()

                if self.controller_step % args.log_step_genome == 0:
                    self._summarize_best_genome(genomes_all)
                    genomes_all = []

                # check for stopping
                elapsed_time = time.time() - self.start_time
                if args.max_time is not None:
                    if elapsed_time >= args.max_time:
                        logger.info('Stopping b/c max time exceeded: {}'.format(elapsed_time))
                        return True
                else:
                    n_gens = int(np.round(self.n_models / args.log_step_genome))
                    if n_gens >= args.max_generations + 1:
                        logger.info('Stopping b/c max_generations: {}/{}. Run time: {}'.format(n_gens,
                                                                                               args.max_generations, elapsed_time))
                        return True

        logger.info('Controller finished training within time: {}'.format(elapsed_time))
        return True

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        logger.info(
            f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)

    def _summarize_best_genome(self, genomes_all):

        logger.info('*' * 50)
        logger.info('n_models: {}, best_fitness: {:.3f}, controller_step: {}'.format(
                    self.n_models, self.best_genome.fitness, self.controller_step))
        logger.info('best_genome:\n{}'.format(self.best_genome.model_string))

        if self.tb is not None:

            # best genome info
            self.tb.scalar_summary(f'best_genome/fitness',
                                    self.best_genome.fitness,
                                    self.controller_step)

            self.tb.text_summary(f'best_genome/model_string',
                                 self.best_genome.model_string,
                                 self.controller_step)

            # save best genome image
            fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                     f'{self.best_genome.fitness:6.4f}-best_genome.png')
            path = os.path.join(args.model_dir, 'networks', fname)

            graph_deep_net(self.amb, dp=self.profile, show_disabled=False,
                           prune=True, genome=self.best_genome,
                           fname=None, save_file=path.replace('.png', ''))

            self.tb.image_summary('best_genome/sample', path, self.controller_step)

            # can't do this in AMB b/c you don't have the model
            self.tb.scalar_summary(f'best_genome/num_params',
                                   utils.num_params(self.best_model),
                                   self.controller_step)

            # generation summary
            fitnesses = [x[0].fitness for x in genomes_all]
            bp_iters = [x[-1] for x in genomes_all]

            self.tb.histogram_summary('generation/fitnesses', fitnesses, self.controller_step)
            self.tb.histogram_summary('generation/bp_iters', bp_iters, self.controller_step)

            self.tb.scalar_summary(f'gneration/mean_bp_iters',
                                   np.mean(bp_iters),
                                   self.controller_step)

            # plot best genomes
            """
            genomes_all.sort(key=lambda x: x[1])
            genomes_all = genomes_all[:5]

            for genome, val_loss, step, bp_iters in genomes_all:
                fname = (f'{self.epoch:03d}-{step:06d}-'
                         f'{val_loss:6.4f}.png')

                path = os.path.join(args.model_dir, 'networks', fname)

                graph_deep_net(self.amb, dp=self.profile, show_disabled=False,
                               prune=True, genome=genome,
                               fname=None, save_file=path.replace('.png', ''))

                self.tb.image_summary('controller/sample', path, step)
            """

    def predict(self, X):
        self.best_model.cuda()
        self.best_model.eval()
        self.best_model.reset_hidden()
        inp = ag.Variable(to.Tensor(X.values)).cuda()
        p, hidden = self.best_model(inp, None)
        p = p.cpu().data.numpy()
        if self.is_class:
            #p = np.argmax(p, axis=1)
             # rounding might not work if there are many classes, so do argmax instead
            maxp = np.zeros_like(p)
            maxp[np.arange(p.shape[0]), p.argmax(axis=1)] = 1
            p = maxp
        return pd.DataFrame(p, index=X.index)

    def predict_proba(self, X):
        self.best_model.cuda()
        self.best_model.eval()
        self.best_model.reset_hidden()
        inp = ag.Variable(to.Tensor(X.values)).cuda()
        p, hidden = self.best_model(inp, None)
        p = p.cpu().data.numpy()
        # return pd.DataFrame(p, columns=X.columns, index=X.index)
        return p


def eval_model(model, dp, df):
    from sklearn.metrics import classification_report, mean_squared_error, r2_score, log_loss, f1_score

    xt, yt = dp.transform(df)
    p = model.predict(xt)

    y_pred = dp.inverse_transform(p)
    y_true = df.loc[:, dp.target]

    is_class = dp.has_categorical_target()

    if is_class:
        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.debug('f1: %.4f', f1)
        try:
            logger.debug(classification_report(y_true, y_pred))
        except:
            pass
        return float(f1)
    else:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logger.debug('MSE: %.3f, r^2: %.3f', mse, r2)
        return float(r2)


def main():

    sys.path.append(os.path.join(settings.ROOT_DIR, 'tests'))
    from datasets import Datasets

    if args.dataset == 'all':
        from compare_models import datasets
    else:
        datasets = [eval(args.dataset)]
    logger.info('Number of datasets: {}'.format(len(datasets)))

    # df_meta = pd.read_csv('/home/jgoode/amb-data/meta-feats.csv',
    #                       index_col='dataset_name').drop(['Unnamed: 0'], axis=1)

    now = utils.get_time()
    model_dir = args.model_dir

    for d in datasets:

        df_train = d.train()
        df_test = d.test()
        target = d.meta().target
        name = d.meta().name

        if args.max_time is not None:
            args.max_generations = None

        dp = DataProfiler(target=target)
        x_train, y_train = dp.fit_transform(df_train)

        if args.model == 'both':
            model_types = ['nas', 'amb']
        else:
            model_types = [args.model]

        test_score = None

        for model_type in model_types:

            args.model_dir = os.path.join(model_dir, '{}-{}-{}'.format(name, model_type.upper(), now))
            os.makedirs(args.model_dir)

            utils.save_args(args)

            if model_type == 'nas':
                
                settings.SHARE_WEIGHTS = True

                model = SimpleNAS(profile=dp)

                start_time = time.time()
                model.fit(x_train, y_train, use_tensorboard=True)
                train_time = time.time() - start_time

                test_score = eval_model(model, dp, df_test)
                logger.info('DONE: model_dir: {}: score: {}'.format(args.model_dir, test_score))

            elif model_type == 'amb':

                settings.SHARE_WEIGHTS = False

                from amb.model.supervised.amune import Amune

                if args.max_generations is not None:
                    # am = AutoModel(max_generation=args.max_generation, show_plots=False)
                    model = Amune(max_generation=args.max_generations,
                                  profile=dp,
                                  skl_genomes=False,
                                  tb_logdir=args.model_dir)
                else:
                    assert args.max_time > 0
                    model = Amune(max_time=args.max_time,
                                  profile=dp,
                                  skl_genomes=False,
                                  tb_logdir=args.model_dir)

                start_time = time.time()
                model.fit(x_train, y_train)
                train_time = time.time() - start_time

                test_score = eval_model(model, dp, df_test)
                logger.info('DONE: model_dir: {}: score: {}'.format(args.model_dir, test_score))

            else:
                raise ValueError('Unknown model: %s' % model_type)

            output = {'data_name': name,
                      'model_type': model_type,
                      'test_score': test_score,
                      'models': model.best_genome.model_string,
                      'outdir': args.model_dir,
                      'train_time': train_time,
                      }

            with open(os.path.join(model_dir, 'results.json'), 'a') as f:
                f.write(json.dumps(output) + '\n')

if __name__ == '__main__':
    main()
