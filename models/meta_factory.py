"""
This is a slim version of the code from https://github.com/SsnL/dataset-distillation
"""

import torch
import torchvision
import logging

import torch.nn as nn
import torch.nn.functional as F
import functools
import math
import types
from contextlib import contextmanager
from torch.optim import lr_scheduler
from six import add_metaclass
from itertools import chain
from copy import deepcopy
from models.helpers import *

class MetaFactory(type):
    def __call__(cls, *args, **kwargs):
        r"""Called when you call ReparamModule(...) """
        net = type.__call__(cls, *args, **kwargs)

        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    pass
                    # logging.warning((
                    #     '{} contains buffer {}. The buffer will be treated as '
                    #     'a constant and assumed not to change during gradient '
                    #     'steps. If this assumption is violated (e.g., '
                    #     'BatcHNorm*d\' running_mean/var), the computation will '
                    #     'be incorrect.').format(m.__class__.__name__, n))

        net._weights_module_names = tuple(w_modules_names)

        # Put to correct device before we do stuff on parameters
        #net = net.to(device)

        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter
        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(MetaFactory)
class ReparamModule(nn.Module):
    """
    Make an architecture inherit this class instead of nn.Module to allow .forward_with_params()
    This changes state_dict() to a one value dict containing 'flat_w'

    This requires self.device to be defined in the module

    """

    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        if clone:
            return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad).to(device=self.device)
        return self.flat_w.to(device=self.device)

    @contextmanager
    def unflatten_weight(self, flat_w):
        """
        This changes self.state_dict()
        from -->  odict_keys(['flat_w'])
        to   -->  odict_keys(['flat_w', 'layers.0.weight', 'layers.0.bias', ... ]

        Somehow removes 'bias=False' in self._weights_module_names conv names, and
        replaces 'bias=False' by 'bias=True' in linear layers.

        type(self.state_dict()) = <class 'collections.OrderedDict'> before and after

        """
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def forward_with_param(self, inp, new_w):
        #print(type(self.state_dict()))
        with self.unflatten_weight(new_w):
            # print('FLATTENED')
            # print('state_dict: ', type(self.state_dict()), [(k, v.shape) for k,v in self.state_dict().items()])
            # print('self._weights_module_names: ', self._weights_module_names)
            return nn.Module.__call__(self, inp)

    def __call__(self, inp):
        return self.forward_with_param(inp, self.flat_w)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Make load_state_dict work on both singleton dicts
        containing a flattened weight tensor and full dicts
        containing unflattened weight tensors. Useful when loading
        weights from non-meta architectures

        """
        if len(state_dict) == 1 and 'flat_w' in state_dict:
            return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        with self.unflatten_weight(self.flat_w):
            flat_w = self.flat_w
            del self.flat_w
            super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        self.register_parameter('flat_w', flat_w)

    def unflattened_weights(self):
        #print(float(torch.sum(self.state_dict()['flat_w'])))

        with self.unflatten_weight(self.flat_w):
            state_dict = deepcopy(self.state_dict())
            del state_dict['flat_w']

        return state_dict

    def layer_names(self):
        layer_names = []
        layer_count = 0
        prev_layer = None

        for (name, n) in zip(self._weights_module_names, self._weights_numels):

            if name[0] != prev_layer:
                layer_count += 1
                prev_layer = name[0]

            if isinstance(name[0], torch.nn.Conv2d) and name[1]=='weight':
                layer_names.append('L{}_conv_W_s{}'.format(layer_count, n))
            elif isinstance(name[0], torch.nn.Conv2d) and name[1]=='bias':
                layer_names.append('L{}_conv_b_s{}'.format(layer_count, n))
            elif isinstance(name[0], torch.nn.BatchNorm2d) and name[1]=='weight':
                layer_names.append('L{}_bn_W_s{}'.format(layer_count, n))
            elif isinstance(name[0], torch.nn.BatchNorm2d) and name[1]=='bias':
                layer_names.append('L{}_bn_b_s{}'.format(layer_count, n))
            elif isinstance(name[0], torch.nn.Linear) and name[1]=='weight':
                layer_names.append('L{}_fc_W_s{}'.format(layer_count, n))
            elif isinstance(name[0], torch.nn.Linear) and name[1]=='bias':
                layer_names.append('L{}_fc_b_s{}'.format(layer_count, n))
            else:
                raise ValueError('Unknown layer type {}'.format(name))


        return layer_names

    def get_bn_masks(self):
        """
        Returns 2 boolean masks of size n_weights,
        where ones correspond to batchnorm gammas in first mask,
        and batchnorm betas in second mask
        """

        gammas_mask = torch.zeros(self.flat_w.shape[0], dtype=torch.bool)
        betas_mask = torch.zeros(self.flat_w.shape[0], dtype=torch.bool)
        i = 0

        for (name, n) in zip(self._weights_module_names, self._weights_numels):
            is_BN = isinstance(name[0], torch.nn.BatchNorm2d) or isinstance(name[0], torch.nn.BatchNorm1d)
            if  is_BN and name[1]=='weight':
                gammas_mask[i:i+n] = 1
            elif is_BN and name[1]=='bias':
                betas_mask[i:i+n] = 1
            i += n

        return gammas_mask, betas_mask

    def flattened_unflattened_weights(self):
        """
        somehow unflattening weights changes the value of their sum.
        This looks like it's because permutation matters in float 32 sum operation and
        so different data structures give different results to the same operations
        even though they contain the same values. Here unflattening and reflattening
        recovers the sum value of the original self.get_param() method.
        """

        with self.unflatten_weight(self.flat_w):
            state_dict = deepcopy(self.state_dict())
            del state_dict['flat_w']

        flat_w = torch.cat([w.reshape(-1) for w in state_dict.values()], 0) #.type(torch.DoubleTensor) doesn't change behaviour
        return flat_w

    def initialize(self, init_type='xavier', init_param=1, init_norm_weights=1, inplace=True):
        if inplace:
            flat_w = self.flat_w
        else:
            flat_w = torch.empty_like(self.flat_w).requires_grad_()

        with torch.no_grad():
            with self.unflatten_weight(flat_w):
                initialize(self, init_type=init_type, init_param=init_param, init_norm_weights=init_norm_weights)

        return flat_w









