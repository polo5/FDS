"""
This is the base code to learn the learning rate, momentum and weight decay
non-greedily with forward mode differentiation, over long horizons (e.g. CIFAR10)
"""

import os
import time
import shutil
import torch
import torch.optim as optim
import pickle

from utils.logger import *
from utils.helpers import *
from utils.datasets import *
from models.selector import *



class MetaLearner(object):
    def __init__(self, args):
        self.args = args

        ## Optimization
        self.hypers_init()
        self.cross_entropy = nn.CrossEntropyLoss()

        ## Experiment Set Up
        self.best_outer_step = 0
        self.best_validation_acc = 0
        ns, learnables = (self.args.n_lrs, self.args.n_moms, self.args.n_wds), (self.args.learn_lr, self.args.learn_mom, self.args.learn_wd)
        self.all_lr_schedules, self.all_mom_schedules, self.all_wd_schedules  = [torch.zeros((self.args.n_outer_steps+1, n)) for n in ns] #+1 since save init schedules and last schedule
        self.all_lr_raw_grads, self.all_mom_raw_grads, self.all_wd_raw_grads = [torch.zeros((self.args.n_outer_steps, n)) if l else None for (n,l) in zip(ns, learnables)]
        self.all_lr_smooth_grads, self.all_mom_smooth_grads, self.all_wd_smooth_grads = [torch.zeros((self.args.n_outer_steps, n)) if l else None for (n,l) in zip(ns, learnables)]

        self.experiment_path = os.path.join(self.args.log_directory_path, self.args.experiment_name)
        self.checkpoint_path = os.path.join(self.experiment_path, 'checkpoint.pth.tar')
        if os.path.exists(self.experiment_path):
            if self.args.use_gpu and os.path.isfile(self.checkpoint_path):
                raise NotImplementedError(f"Experiment folder {self.experiment_path} already exists") #TODO: restore code from ckpt
            else:
                shutil.rmtree(self.experiment_path) # clear debug logs on cpu
                os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)
        copy_file(os.path.realpath(__file__), self.experiment_path) # save this python file in logs folder
        self.logger = Logger(self.experiment_path, 'run_results.csv')

        ## Save and Print Args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')
        print('\nLogging every {} outer_steps and every {} epochs per outer_step\n'.format(self.args.outer_step_log_freq, self.args.epoch_log_freq))

    def hypers_init(self):
        """ initialize hyperparameters """

        self.inner_lrs = self.args.inner_lr_init*torch.ones(self.args.n_lrs, device=self.args.device)
        self.inner_lrs_grad = torch.zeros_like(self.inner_lrs) # lr hypergradient
        self.lr_hypersigns = torch.zeros(self.args.n_lrs, device=self.args.device)
        self.lr_step_sizes = self.args.lr_init_step_size*torch.ones(self.args.n_lrs, device=self.args.device)

        self.inner_moms = self.args.inner_mom_init*torch.ones(self.args.n_moms, device=self.args.device)
        self.inner_moms_grad = torch.zeros_like(self.inner_moms)
        self.mom_hypersigns = torch.zeros(self.args.n_moms, device=self.args.device)
        self.mom_step_sizes = self.args.mom_init_step_size*torch.ones(self.args.n_moms, device=self.args.device)

        self.inner_wds = self.args.inner_wd_init*torch.ones(self.args.n_wds, device=self.args.device)
        self.inner_wds_grad = torch.zeros_like(self.inner_wds)
        self.wd_hypersigns = torch.zeros(self.args.n_wds, device=self.args.device)
        self.wd_step_sizes = self.args.wd_init_step_size*torch.ones(self.args.n_wds, device=self.args.device)

    def get_hypers(self, epoch, batch_idx):
        """return hyperparameters to be used for given batch"""

        lr_index = int(self.args.n_lrs * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches_for_this_outer_step)
        lr = float(self.inner_lrs[lr_index])

        mom_index = int(self.args.n_moms * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches_for_this_outer_step)
        mom = float(self.inner_moms[mom_index])

        wd_index = int(self.args.n_wds * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches_for_this_outer_step)
        wd = float(self.inner_wds[wd_index])

        return lr, mom, wd, lr_index, mom_index, wd_index

    def to_prune(self, epoch, batch_idx, n_hypers):
        """ Do we skip calculation of Z for this batch?"""

        if self.args.pruning_ratio==0:
            to_prune=False
        else:
            n_batches_per_hyper = int(self.n_total_batches_for_this_outer_step/n_hypers)
            current_global_batch_idx = epoch*self.n_batches_per_epoch + batch_idx
            current_global_batch_idx_per_hyper = current_global_batch_idx % n_batches_per_hyper

            if self.args.pruning_mode=='alternate': #rounded to nearest integer, so r=0.25 -> prune 1 in 4 but r=0.21 -> 1 in 4 also
                if self.args.pruning_ratio>=0.5: #at least 1 in 2 pruned
                    keep_freq = int(1/(1-self.args.pruning_ratio))
                    to_prune = (current_global_batch_idx_per_hyper % keep_freq != 0)
                else:
                    prune_freq = int(1/(self.args.pruning_ratio))
                    to_prune = (current_global_batch_idx_per_hyper % prune_freq == 0)

            elif self.args.pruning_mode=='truncate':
                to_prune = current_global_batch_idx_per_hyper < self.args.pruning_ratio*n_batches_per_hyper

        return to_prune

    def inner_loop(self):
        """
        Compute Z for each hyperparameter to learn over all epochs in the run
        """

        ## Network
        self.classifier = select_model(True, self.args.dataset, self.args.architecture,
                                       self.args.init_type, self.args.init_param,
                                       self.args.device).to(self.args.device)
        self.classifier.train()
        self.weights = self.classifier.get_param()
        velocity = torch.zeros(self.weights.numel(), requires_grad=False, device=self.args.device)

        ## Forward Mode Init
        if self.args.learn_lr:
            self.n_batches_per_lr = 0
            Z_lr = torch.zeros((self.weights.numel(), self.args.n_lrs), device=self.args.device)
            C_lr = torch.zeros((self.weights.numel(), self.args.n_lrs), device=self.args.device)
        else:
            Z_lr = None

        if self.args.learn_mom:
            self.n_batches_per_mom = 0
            Z_mom = torch.zeros((self.weights.numel(), self.args.n_moms), device=self.args.device)
            C_mom = torch.zeros((self.weights.numel(), self.args.n_moms), device=self.args.device)
        else:
            Z_mom = None

        if self.args.learn_wd:
            self.n_batches_per_wd = 0
            Z_wd = torch.zeros((self.weights.numel(), self.args.n_wds), device=self.args.device)
            C_wd = torch.zeros((self.weights.numel(), self.args.n_wds), device=self.args.device)
        else:
            Z_wd = None

        ## Inner Loop Over All Epochs
        for epoch in range(self.n_inner_epochs_for_this_outer_step):
            t0_epoch = time.time()

            for batch_idx, (x_train, y_train) in enumerate(self.train_loader):
                lr, mom, wd, lr_index, mom_index, wd_index = self.get_hypers(epoch, batch_idx)
                #print(f'epoch {epoch} batch {batch_idx} -- lr idx {lr_index} -- mom idx {mom_index} -- wd index {wd_index}')
                x_train, y_train = x_train.to(device=self.args.device), y_train.to(device=self.args.device)
                train_logits = self.classifier.forward_with_param(x_train, self.weights)
                train_loss = self.cross_entropy(train_logits, y_train)
                grads = torch.autograd.grad(train_loss, self.weights, create_graph=True)[0]
                if self.args.clamp_grads: grads.clamp_(-self.args.clamp_grads_range, self.args.clamp_grads_range)

                if self.args.learn_lr and not self.to_prune(epoch, batch_idx, self.args.n_lrs):
                    #print('update lr')
                    self.n_batches_per_lr += 1
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_lrs),device=self.args.device)
                    for j in range(lr_index + 1):
                        retain = (j != lr_index) or self.args.learn_mom or self.args.learn_wd
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_lr[:, j], self.weights, retain_graph=retain)[0]

                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = Z_lr*(1 - lr*wd) - lr*H_times_Z
                    B = - mom*lr*C_lr
                    B[:,lr_index] -= grads.detach() + wd*self.weights.detach() + mom*velocity
                    C_lr = mom*C_lr + H_times_Z + wd*Z_lr

                    Z_lr = A_times_Z + B

                if self.args.learn_mom and not self.to_prune(epoch, batch_idx, self.args.n_moms):
                    #print('update mom')
                    self.n_batches_per_mom += 1
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_moms),device=self.args.device)
                    for j in range(mom_index + 1):
                        retain = (j != mom_index) or self.args.learn_wd
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_mom[:, j], self.weights, retain_graph=retain)[0]

                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = (1 - lr*wd)*Z_mom - lr*H_times_Z
                    B = -lr*mom*C_mom
                    B[:, mom_index] -= lr*velocity
                    C_mom = mom*C_mom + H_times_Z + wd * Z_mom
                    C_mom[:, mom_index] += velocity

                    Z_mom = A_times_Z + B

                if self.args.learn_wd and not self.to_prune(epoch, batch_idx, self.args.n_wds):
                    #print('update wd')
                    self.n_batches_per_wd += 1
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_wds),device=self.args.device)
                    for j in range(wd_index + 1):
                        retain = (j != wd_index)
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_wd[:, j], self.weights, retain_graph=retain)[0]

                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = (1 - lr*wd)*Z_wd - lr*H_times_Z
                    B = - lr*mom*C_wd
                    B[:, wd_index] -= lr*self.weights.detach()
                    C_wd = mom*C_wd + H_times_Z + wd*Z_wd
                    C_wd[:, wd_index] += self.weights.detach()

                    Z_wd = A_times_Z + B

                ## SGD inner update
                self.weights.detach_(), grads.detach_()
                velocity = velocity*mom + (grads + wd*self.weights)
                self.weights = self.weights - lr*velocity
                self.weights.requires_grad_()

            print(f'--- Ran epoch {epoch+1} in {format_time(time.time()-t0_epoch)} ---')

        if self.args.learn_lr: self.n_batches_per_lr /= self.args.n_lrs # each hyper gets same # of updates regardless of pruning mode
        if self.args.learn_mom: self.n_batches_per_mom /= self.args.n_moms
        if self.args.learn_wd: self.n_batches_per_wd /= self.args.n_wds


        return Z_lr, Z_mom, Z_wd

    def outer_step(self, outer_step_idx, Z_lr_final, Z_mom_final, Z_wd_final):
        """
        Calculate hypergradients and update hyperparameters accordingly.
        """

        ## Calculate validation gradients with final weights of inner loop
        self.running_val_grad = AggregateTensor()
        for batch_idx, (x_val, y_val) in enumerate(self.val_loader): #need as big batches as train mode for BN train mode
            x_val, y_val = x_val.to(device=self.args.device), y_val.to(device=self.args.device)
            val_logits = self.classifier.forward_with_param(x_val, self.weights)
            val_loss = self.cross_entropy(val_logits, y_val)
            dLval_dw = torch.autograd.grad(val_loss, self.weights)[0]
            self.running_val_grad.update(dLval_dw)

        ## Update hyperparams
        print('')
        if self.args.learn_lr:
            self.inner_lrs_grad = self.running_val_grad.avg() @ Z_lr_final / self.n_batches_per_lr
            self.all_lr_raw_grads[outer_step_idx] = self.inner_lrs_grad.detach()
            print('RAW LR GRADS: ', ["{:.2E}".format(float(i)) for i in self.inner_lrs_grad])
            new_hypersigns = torch.sign(self.inner_lrs_grad) #Nans and zero have sign 0
            flipped_signs = self.lr_hypersigns*new_hypersigns # 1, -1 or 0
            multipliers = torch.tensor([self.args.lr_step_decay if f==-1.0 else 1.0 for f in flipped_signs], device=self.args.device)
            self.lr_step_sizes = multipliers*self.lr_step_sizes
            self.lr_hypersigns = new_hypersigns
            deltas = new_hypersigns*self.lr_step_sizes # how much to change hyperparameter by
            self.lr_converged = ((self.lr_step_sizes/self.inner_lrs) < self.args.converged_frac).all()
            self.inner_lrs = self.inner_lrs - deltas
            self.all_lr_smooth_grads[outer_step_idx] = deltas
            print('SMOOTH LR DELTAS: ', ["{:02.2f}".format(float(i)) for i in deltas])

        
        if self.args.learn_mom:
            self.inner_moms_grad = self.running_val_grad.avg() @ Z_mom_final / self.n_batches_per_mom
            self.all_mom_raw_grads[outer_step_idx] = self.inner_moms_grad.detach()
            print('RAW MOM GRADS: ', ["{:.2E}".format(float(i)) for i in self.inner_moms_grad])
            new_hypersigns = torch.sign(self.inner_moms_grad) #Nans and zero have sign 0
            flipped_signs = self.mom_hypersigns*new_hypersigns # 1, -1 or 0
            multipliers = torch.tensor([self.args.mom_step_decay if f==-1.0 else 1.0 for f in flipped_signs], device=self.args.device)
            self.mom_step_sizes = multipliers*self.mom_step_sizes
            self.mom_hypersigns = new_hypersigns
            deltas = new_hypersigns*self.mom_step_sizes # how much to change hyperparameter by
            self.mom_converged = ((self.mom_step_sizes/self.inner_moms) < self.args.converged_frac).all()
            self.inner_moms = self.inner_moms - deltas
            self.all_mom_smooth_grads[outer_step_idx] = deltas
            print('SMOOTH MOM DELTAS: ', ["{:02.2f}".format(float(i)) for i in deltas])
        
        if self.args.learn_wd:
            self.inner_wds_grad = self.running_val_grad.avg() @ Z_wd_final / self.n_batches_per_wd
            self.all_wd_raw_grads[outer_step_idx] = self.inner_wds_grad.detach()
            print('RAW WD GRADS: ', ["{:.2E}".format(float(i)) for i in self.inner_wds_grad])
            new_hypersigns = torch.sign(self.inner_wds_grad) #Nans and zero have sign 0
            flipped_signs = self.wd_hypersigns*new_hypersigns # 1, -1 or 0
            multipliers = torch.tensor([self.args.wd_step_decay if f==-1.0 else 1.0 for f in flipped_signs], device=self.args.device)
            self.wd_step_sizes = multipliers*self.wd_step_sizes
            self.wd_hypersigns = new_hypersigns
            deltas = new_hypersigns*self.wd_step_sizes # how much to change hyperparameter by
            self.wd_converged = ((self.wd_step_sizes/self.inner_wds) < self.args.converged_frac).all()
            self.inner_wds = self.inner_wds - deltas
            self.all_wd_smooth_grads[outer_step_idx] = deltas
            print('SMOOTH WD DELTAS: ', ["{:02.2f}".format(float(i)) for i in deltas])

        self.converged = (self.lr_converged if self.args.learn_lr else True) and (self.mom_converged if self.args.learn_mom else True) and (self.wd_converged if self.args.learn_wd else True)
         
    def run(self):
        """ Run meta learning experiment """

        t0 = time.time()

        for outer_step_idx in range(self.args.n_outer_steps): # number of outer steps

            ## Set up
            self.n_inner_epochs_for_this_outer_step = self.args.n_inner_epochs_per_outer_steps[outer_step_idx]
            print(f'\nOuter step {outer_step_idx+1}/{self.args.n_outer_steps} --- current budget of {self.n_inner_epochs_for_this_outer_step} epochs --- using:')
            print('lrs = ', [float('{:02.2e}'.format(el)) for el in self.inner_lrs],
                  'moms = ', [float('{:02.2e}'.format(el)) for el in self.inner_moms],
                  'wds = ', [float('{:02.2e}'.format(el)) for el in self.inner_wds])
            self.all_lr_schedules[outer_step_idx], self.all_mom_schedules[outer_step_idx], self.all_wd_schedules[outer_step_idx] = self.inner_lrs.detach(), self.inner_moms.detach(), self.inner_wds.detach()
            self.save_state(outer_step_idx) # state and lrs saved correspond to those set at the beginning of the outer_step


            ## New data split for each outer_step
            self.train_loader, self.val_loader, self.test_loader  = get_loaders(datasets_path=self.args.datasets_path,
                                                                                dataset=self.args.dataset,
                                                                                train_batch_size=self.args.train_batch_size,
                                                                                val_batch_size=self.args.val_batch_size,
                                                                                val_source='train',
                                                                                val_train_fraction=self.args.val_train_fraction,
                                                                                val_train_overlap=self.args.val_train_overlap,
                                                                                workers=self.args.workers,
                                                                                train_infinite=False,
                                                                                val_infinite=False,
                                                                                cutout=self.args.cutout,
                                                                                cutout_length=self.args.cutout_length,
                                                                                cutout_prob=self.args.cutout_prob)

            self.n_batches_per_epoch = len(self.train_loader)
            self.n_total_batches_for_this_outer_step = self.n_inner_epochs_for_this_outer_step * self.n_batches_per_epoch
            ## Update Hypers
            Z_lr_final, Z_mom_final, Z_wd_final = self.inner_loop()
            self.outer_step(outer_step_idx, Z_lr_final, Z_mom_final, Z_wd_final)

            ## See if schedule used for this outer_step led to best validation
            _, val_acc = self.validate(self.weights)
            _, test_acc = self.test(self.weights)
            if val_acc > self.best_validation_acc:
                self.best_validation_acc = val_acc
                self.best_outer_step = outer_step_idx
                #print(f'Best validation acc at outer_step idx {outer_step_idx}')

            ## Break if all hyperparameters have converged
            if self.converged:
                print('STOP HYPERTRAINING BECAUSE ALL HYPERPARAMETERS HAVE CONVERGED')
                break

            ## Time
            time_so_far = time.time() - t0
            self.logger.write({'budget': self.n_inner_epochs_for_this_outer_step, 'time': time_so_far,
                               'val_acc': val_acc, 'test_acc': test_acc})
        print(f'final val acc {100*val_acc:.2g} -- final test_acc: {100*test_acc:.2g}')

        ## Logging Final Metrics
        self.all_lr_schedules[outer_step_idx+1], self.all_mom_schedules[outer_step_idx+1], self.all_wd_schedules[outer_step_idx+1] = self.inner_lrs.detach(), self.inner_moms.detach(), self.inner_wds.detach() #last schedule was never trained on
        self.save_state(outer_step_idx+1)
        avg_test_loss, avg_test_acc = self.test(self.weights)

        return avg_test_acc

    def validate(self, weights, fraction=1.0):
        """ Fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(self.val_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier.forward_with_param(x, weights)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.val_loader):
                    break

        self.classifier.train()
        return float(running_loss.avg()), float(running_acc.avg())

    def test(self, weights, fraction=1.0):
        """ Fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier.forward_with_param(x, weights)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.test_loader):
                    break

        self.classifier.train()
        return float(running_loss.avg()), float(running_acc.avg())

    def save_state(self, outer_step_idx):

        torch.save({'args': self.args,
                    'outer_step_idx': outer_step_idx,
                    'best_outer_step': self.best_outer_step,
                    'best_validation_acc': self.best_validation_acc,
                    'all_lr_schedules': self.all_lr_schedules,
                    'all_lr_raw_grads': self.all_lr_raw_grads,
                    'all_lr_smooth_grads': self.all_lr_smooth_grads,
                    'all_mom_schedules': self.all_mom_schedules,
                    'all_mom_raw_grads': self.all_mom_raw_grads,
                    'all_mom_smooth_grads': self.all_mom_smooth_grads,
                    'all_wd_schedules': self.all_wd_schedules,
                    'all_wd_raw_grads': self.all_wd_raw_grads,
                    'all_wd_smooth_grads': self.all_wd_smooth_grads}, self.checkpoint_path)


class BaseLearner(object):
    """
    Retrain from scratch using learned schedule and
    whole training set
    """
    def __init__(self, args, lr_schedule, mom_schedule, wd_schedule, log_name):
        self.args = args
        self.inner_lrs = lr_schedule
        self.inner_moms = mom_schedule
        self.inner_wds = wd_schedule

        ## Loaders
        self.args.val_source = 'test' # retrain on full train set from scratch
        self.train_loader, _, self.test_loader  = get_loaders(datasets_path=self.args.datasets_path,
                                                              dataset=self.args.dataset,
                                                              train_batch_size=self.args.train_batch_size,
                                                              val_batch_size=self.args.val_batch_size,
                                                              val_source=self.args.val_source,
                                                              val_train_fraction=self.args.val_train_fraction,
                                                              val_train_overlap=self.args.val_train_overlap,
                                                              workers=self.args.workers,
                                                              train_infinite=False,
                                                              val_infinite=False,
                                                              cutout=self.args.cutout,
                                                              cutout_length=self.args.cutout_length,
                                                              cutout_prob=self.args.cutout_prob)

        self.n_batches_per_epoch = len(self.train_loader)
        self.n_total_batches = self.args.retrain_n_epochs * self.n_batches_per_epoch

        ## Optimizer
        self.classifier = select_model(False, self.args.dataset, self.args.architecture,
                               self.args.init_type, self.args.init_param,
                               self.args.device).to(self.args.device)
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.0, momentum=0.0, weight_decay=0.0) #set hypers manually later
        self.cross_entropy = nn.CrossEntropyLoss()

        ### Set up
        self.experiment_path = os.path.join(args.log_directory_path, args.experiment_name)
        self.logger = Logger(self.experiment_path, log_name)

    def log_init(self):
        self.running_train_loss, self.running_train_acc = AggregateTensor(), AggregateTensor()

    def log(self, epoch, avg_train_loss, avg_train_acc):
        avg_test_loss, avg_test_acc = self.test(fraction=0.1 if epoch!=self.args.retrain_n_epochs-1 else 1)
        print('Retrain epoch {}/{} --- Train Acc: {:02.2f}% -- Test Acc: {:02.2f}%'.format(epoch+1, self.args.retrain_n_epochs, avg_train_acc * 100, avg_test_acc * 100))
        self.logger.write({'train_loss': avg_train_loss, 'train_acc': avg_train_acc, 'test_loss': avg_test_loss, 'test_acc': avg_test_acc})
        self.log_init()

    def get_hypers(self, epoch, batch_idx):
        """return hyperparameters to be used for given batch"""

        lr_index = int(self.args.n_lrs * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches)
        lr = float(self.inner_lrs[lr_index])

        mom_index = int(self.args.n_moms * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches)
        mom = float(self.inner_moms[mom_index])

        wd_index = int(self.args.n_wds * (epoch*self.n_batches_per_epoch + batch_idx)/self.n_total_batches)
        wd = float(self.inner_wds[wd_index])

        return lr, mom, wd, lr_index, mom_index, wd_index

    def set_hypers(self, epoch, batch_idx):
        lr, mom, wd, lr_index, mom_index, wd_index = self.get_hypers(epoch, batch_idx)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom
            param_group['weight_decay'] = wd

        #print(f'Setting: lr={lr}, mom={mom}, wd={wd}')

    def run(self):

        for epoch in range(self.args.retrain_n_epochs):
            avg_train_loss, avg_train_acc = self.train(epoch)
            self.log(epoch, avg_train_loss, avg_train_acc)

        test_loss, test_acc = self.test()
        return test_acc

    def train(self, epoch):
        self.classifier.train()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        for batch_idx, (x,y) in enumerate(self.train_loader):
            self.set_hypers(epoch, batch_idx)
            x, y = x.to(device=self.args.device), y.to(device=self.args.device)
            logits = self.classifier(x)
            loss = self.cross_entropy(input=logits, target=y)
            acc1 = accuracy(logits.data, y, topk=(1,))[0]
            running_loss.update(loss, x.shape[0])
            running_acc.update(acc1, x.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(running_loss.avg()), float(running_acc.avg())

    def test(self, fraction=1.0):
        """ fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier(x)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.test_loader):
                    break

        self.classifier.train()
        return float(running_loss.avg()), float(running_acc.avg())


# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________

def make_experiment_name(args):
    """
    Warning: Windows can have a weird behaviour for long filenames.
    Protip: switch to Ubuntu ;)
    """

    ## Main
    nepr = ''.join([str(i) for i in args.n_inner_epochs_per_outer_steps])
    experiment_name = f'FSL_{args.dataset}_{args.architecture}_nepr{nepr}'
    experiment_name += f'_init{args.init_type}-{args.init_param}'
    experiment_name += f'_tbs{args.train_batch_size}'
    if args.cutout: experiment_name += f'_cutout-p{args.cutout_prob}'
    if args.clamp_HZ: experiment_name += f'_HZclamp{args.clamp_HZ_range}'

    experiment_name += f'_S{args.seed}'

    return experiment_name


def main(args):
    set_torch_seeds(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    meta_learner = MetaLearner(args)
    meta_test_acc = meta_learner.run()
    total_time = time.time() - t0

    to_print = '\n\nMETA TEST ACC: {:02.2f}%'.format(meta_test_acc*100)
    file_name = "final_meta_test_acc_{:02.2f}_total_time_{}".format(meta_test_acc*100, format_time(total_time))
    create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))

    if args.retrain_from_scratch:
        ## Fetch schedules
        # best_idx = meta_learner.best_outer_step
        final_lr_schedule, final_mom_schedule, final_wd_schedule = meta_learner.all_lr_schedules[-1], meta_learner.all_mom_schedules[-1], meta_learner.all_wd_schedules[-1]
        # best_lr_schedule, best_mom_schedule, best_wd_schedule = meta_learner.all_lr_schedules[best_idx], meta_learner.all_mom_schedules[best_idx], meta_learner.all_wd_schedules[best_idx]
        del meta_learner

        ## Retrain Last
        print(f'\n\n\n---------- RETRAINING FROM SCRATCH WITH LAST SCHEDULE (idx {args.n_outer_steps}) ----------')
        print(f'lrs = {final_lr_schedule.tolist()}')
        print(f'moms = {final_mom_schedule.tolist()}')
        print(f'wds = {final_wd_schedule.tolist()}')

        log_name = f'Rerun_last_outer_step.csv'
        base_learner = BaseLearner(args, final_lr_schedule, final_mom_schedule, final_wd_schedule, log_name)
        if args.use_gpu: torch.cuda.empty_cache()
        base_test_acc = base_learner.run()
        to_print += '\nRE-RUN LAST SCHEDULE TEST ACC: {:02.2f}%'.format(base_test_acc*100)
        file_name = "Rerun_last_test_acc_{:02.2f}".format(base_test_acc*100)
        create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))


        # ## Retrain Best Val
        # print(f'\n\n\n---------- RETRAINING FROM SCRATCH WITH BEST VAL SCHEDULE (idx {best_idx}) ----------')
        # print(f'lrs = {best_lr_schedule.tolist()}')
        # print(f'moms = {best_mom_schedule.tolist()}')
        # print(f'wds = {best_wd_schedule.tolist()}')
        #
        # log_name = f'Rerun_best_outer_step_idx_{best_idx}.csv'
        # base_learner = BaseLearner(args, best_lr_schedule, best_mom_schedule, best_wd_schedule, log_name)
        # if args.use_gpu: torch.cuda.empty_cache()
        # base_test_acc = base_learner.run()
        # to_print += '\nRE-RUN BEST SCHEDULE TEST ACC: {:02.2f}%'.format(base_test_acc*100)
        # file_name = "Rerun_best_test_acc_{:02.2f}".format(base_test_acc*100)
        # create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))

    print(to_print)


if __name__ == "__main__":
    import argparse
    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to GreedyGrad')

    ## Main
    parser.add_argument('--learn_lr', type=str2bool, default=True)
    parser.add_argument('--learn_mom', type=str2bool, default=True)
    parser.add_argument('--learn_wd', type=str2bool, default=True)
    parser.add_argument('--n_lrs', type=int, default=7)
    parser.add_argument('--n_moms', type=int, default=1)
    parser.add_argument('--n_wds', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--n_inner_epochs_per_outer_steps', nargs='*', type=int, default=[1, 10, 10, 10, 10, 10, 10, 10, 10, 10], help='number of epochs to run for each outer step')
    parser.add_argument('--pruning_mode', type=str, choices=['alternate', 'truncate'], default='alternate')
    parser.add_argument('--pruning_ratio', type=float, default=0.0, help='<1, how many inner steps to skip Z calculation for expressed as a fraction of total inner steps per hyper')
    ## Architecture
    parser.add_argument('--architecture', type=str, default='WRN-16-1')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal', 'zero', 'default'], help='network initialization scheme')
    parser.add_argument('--init_param', type=float, default=1, help='network initialization param: gain, std, etc.')
    parser.add_argument('--init_norm_weights', type=float, default=1, help='init gammas of BN')
    ## Inner Loop
    parser.add_argument('--inner_lr_init', type=float, default=0, help='SGD inner learning rate init')
    parser.add_argument('--inner_mom_init', type=float, default=0, help='SGD inner momentum init')
    parser.add_argument('--inner_wd_init', type=float, default=0, help='SGD inner weight decay init')
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--clamp_grads', type=str2bool, default=True)
    parser.add_argument('--clamp_grads_range', type=float, default=3, help='clamp inner grads for each batch to +/- that')
    parser.add_argument('--cutout', type=str2bool, default=False)
    parser.add_argument('--cutout_length', type=int, default=16)
    parser.add_argument('--cutout_prob', type=float, default=1, help='clamp inner grads for each batch to +/- that')
    ## Outer Loop
    parser.add_argument('--val_batch_size', type=int, default=500)
    parser.add_argument('--val_train_fraction', type=float, default=0.05)
    parser.add_argument('--val_train_overlap', type=str2bool, default=False, help='if True and val_source=train, val images are also in train set')
    parser.add_argument('--lr_init_step_size', type=float, default=0.1, help='at each iteration grads changed so that each hyper can only change by this fraction (ignoring outer momentum)')
    parser.add_argument('--mom_init_step_size', type=float, default=0.1)
    parser.add_argument('--wd_init_step_size', type=float, default=3e-4)
    parser.add_argument('--lr_step_decay', type=float, default=0.5, help='step size multiplied by this much if hypergrad sign changes')
    parser.add_argument('--mom_step_decay', type=float, default=0.5, help='step size multiplied by this much if hypergrad sign changes')
    parser.add_argument('--wd_step_decay', type=float, default=0.5, help='step size multiplied by this much if hypergrad sign changes')
    parser.add_argument('--clamp_HZ', type=str2bool, default=True)
    parser.add_argument('--clamp_HZ_range', type=float, default=1, help='clamp to +/- that')
    parser.add_argument('--converged_frac', type=float, default=0.05, help='if steps are smaller than this percentage of hypers, stop experiment')
    ## Other
    parser.add_argument('--retrain_from_scratch', type=str2bool, default=True, help='retrain from scratch with learned lr schedule')
    parser.add_argument('--retrain_n_epochs', type=int, default=50, help='interpolates from learned schedule, -1 for same as n_inner_epochs_per_outer_steps[-1]')
    parser.add_argument('--datasets_path', type=str, default="~/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="./logs/")
    parser.add_argument('--epoch_log_freq', type=int, default=1, help='every how many epochs to save summaries')
    parser.add_argument('--outer_step_log_freq', type=int, default=1, help='every how many outer_steps to save the whole run')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    args = parser.parse_args()

    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    assert args.lr_step_decay < 1
    assert args.mom_step_decay < 1
    assert args.wd_step_decay < 1
    assert args.converged_frac < 1
    if args.retrain_n_epochs < 0: args.retrain_n_epochs = args.n_inner_epochs_per_outer_steps[-1]
    assert args.pruning_ratio <= 1
    args.n_outer_steps = len(args.n_inner_epochs_per_outer_steps)

    args.experiment_name = make_experiment_name(args)

    print('\nRunning on device: {}'.format(args.device))
    if args.use_gpu: print(torch.cuda.get_device_name(0))


    main(args)


