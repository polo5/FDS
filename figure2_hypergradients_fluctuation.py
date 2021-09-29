"""
Here we measure hypergradients for several runs when perturbing
 the training data and weight initialization. This must be done on toy
 datasets where reverse-mode differentiation is tractable. This corresponds
 to figure 2 in the paper.
"""

import torch.optim as optim
import pickle

import os
import warnings
import sys
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optimw

from utils.helpers import *
from utils.datasets import *
from models.selector import *


class HyperGradFluctuation(object):
    def __init__(self, args):
        self.args = args
        self.hypergrads_all = torch.zeros((self.args.n_runs, self.args.T))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.init_lr_schedule()

        ## Loaders
        self.infinite_train_loader, self.val_loader, _ = get_loaders(datasets_path=self.args.datasets_path,
                                                                     dataset=self.args.dataset,
                                                                     train_batch_size=self.args.train_batch_size,
                                                                     val_batch_size=self.args.n_val_images,
                                                                     val_source='test',
                                                                     workers=self.args.workers,
                                                                     train_infinite=True,
                                                                     val_infinite=False)
        for x,y in self.val_loader: self.X_val, self.Y_val = x.to(device=self.args.device), y.to(device=self.args.device)

        ## Set up experiment folder
        self.experiment_path = os.path.join(self.args.log_directory_path, self.args.experiment_name)
        if os.path.isfile(os.path.join(self.experiment_path, 'hypergrads.pth.tar')):
            if args.use_gpu: raise FileExistsError(f'Experiment already ran and exists at {self.experiment_path}. \nStopping now')
        else:
            if os.path.exists(self.experiment_path):
                shutil.rmtree(self.experiment_path)
            os.makedirs(self.experiment_path)

        ## Save and Print Args
        copy_file(os.path.realpath(__file__), self.experiment_path)  # save this python file in logs folder
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

    def init_lr_schedule(self):
        if self.args.inner_lr_cosine_anneal:
            dummy_opt = optim.SGD([torch.ones([1], requires_grad=True)], lr=self.args.inner_lr_init)
            dummy_scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=self.args.T)
            lrs = []
            for i in range(self.args.T):
                lrs.append(dummy_scheduler.get_last_lr()[0])
                dummy_opt.step()
                dummy_scheduler.step()
            self.inner_lrs = torch.tensor(lrs, requires_grad=True, device=self.args.device)
        else:
            self.inner_lrs = torch.full((self.args.T,), self.args.inner_lr_init, requires_grad=True, device=self.args.device)

    def inner_and_outer_loop(self):
        for self.inner_step_idx, (x_train, y_train) in enumerate(self.infinite_train_loader):
            x_train, y_train = x_train.to(self.args.device, self.args.dtype), y_train.to(self.args.device)
            train_logits = self.classifier.forward_with_param(x_train, self.weights)
            train_loss = self.cross_entropy(train_logits, y_train)

            grads = torch.autograd.grad(train_loss, self.weights, create_graph=True)[0]
            if self.args.clamp_inner_grads: grads.clamp_(-self.args.clamp_inner_grads_range, self.args.clamp_inner_grads_range)
            self.velocity = self.args.inner_momentum * self.velocity + (grads + self.args.inner_weight_decay * self.weights)
            self.weights = self.weights - self.inner_lrs[self.inner_step_idx] * self.velocity

            if self.args.greedy:
                self.compute_hypergradients() #only populates .grad of one item in self.inner_lrs
                self.weights.detach_().requires_grad_()
                self.velocity.detach_().requires_grad_()

            if self.inner_step_idx+1 == self.args.T: break

        if not self.args.greedy: self.compute_hypergradients() #populates .grad of all items in self.inner_lrs

    def compute_hypergradients(self):
        val_logits = self.classifier.forward_with_param(self.X_val, self.weights)
        val_loss = self.cross_entropy(val_logits, self.Y_val)
        val_loss.backward()

    def run(self):
        for self.run_idx in range(self.args.n_runs):
            self.classifier = select_model(True, self.args.dataset, self.args.architecture,
                                           self.args.init_type, self.args.init_param,
                                           self.args.device).to(self.args.device)
            self.weights = self.classifier.get_param()
            self.velocity = torch.zeros(self.weights.numel(), device=self.args.device)
            self.inner_and_outer_loop()
            self.hypergrads_all[self.run_idx] = self.inner_lrs.grad.detach()
            self.inner_lrs.grad.data.zero_()

        self.save_final()

    def save_final(self):
        torch.save({'args': self.args,
                    'hypergrads_all': self.hypergrads_all},
                   os.path.join(self.experiment_path, 'hypergrads.pth.tar'))
        print(f"Saved hypergrads to {os.path.join(self.experiment_path, 'hypergrads.pth.tar')}")


# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________


def make_experiment_name(args):
    experiment_name = f'Hg_{args.dataset}_{args.init_type}_T{args.T}_tbs{args.train_batch_size}_mom{args.inner_momentum}_wd{args.inner_weight_decay}_ilr{args.inner_lr_init}'
    if args.inner_lr_cosine_anneal: experiment_name += f'cosine'
    if args.greedy: experiment_name += f'_GREEDY'
    if args.dtype == torch.float64: experiment_name += '_FL64'
    experiment_name += f'_S{args.seed}'

    return experiment_name


def main(args):
    set_torch_seeds(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    hypervariance_learner = HyperGradFluctuation(args)
    hypervariance_learner.run()
    total_time = time.time() - t0

    with open(os.path.join(args.log_directory_path, args.experiment_name, 'TOTAL_TIME_' + format_time(total_time)), 'w+') as f:
        f.write("NA")


if __name__ == "__main__":
    import argparse

    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to GreedyGrad')

    ## Main
    parser.add_argument('--T', type=int, default=250, help='number of batches for the task and to learn a schedule over')
    parser.add_argument('--n_runs', type=int, default=100, help='how many times to compute hypergrads, with different train-val-split each time')
    parser.add_argument('--dataset', type=str, default='SVHN')
    parser.add_argument('--greedy', type=str2bool, default=False)
    parser.add_argument('--architecture', type=str, default='LeNet')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal', 'zero', 'default'], help='network initialization scheme')
    parser.add_argument('--init_param', type=float, default=1, help='network initialization param: gain, std, etc.')
    parser.add_argument('--n_val_images', type=int, default=2000, help='ignored unless val_source=train') #20% of 60k=12000
    ## Inner Loop
    parser.add_argument('--inner_lr_init', type=float, default=0.01, help='Used to initialize inner learning rate(s).')
    parser.add_argument('--inner_lr_cosine_anneal', type=str2bool, default=True, help='Initial schedule is cosine annealing')
    parser.add_argument('--inner_momentum', type=float, default=0.9, help='SGD inner momentum')
    parser.add_argument('--inner_weight_decay', type=float, default=0.0, help='SGD + ADAM inner weight decay')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--clamp_inner_grads', type=str2bool, default=True)
    parser.add_argument('--clamp_inner_grads_range', type=float, default=1, help='clamp inner grads for each batch to +/- that')
    ## Misc
    parser.add_argument('--datasets_path', type=str, default="~/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="./logs/")
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    args = parser.parse_args()

    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    if args.dtype == 'float64':
        torch.set_default_tensor_type(torch.DoubleTensor)  # changes weights and tensors but not loaders
    args.dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    print('\nRunning on device: {}'.format(args.device))

    args.experiment_name = make_experiment_name(args)
    main(args)


