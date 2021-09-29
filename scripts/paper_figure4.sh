#!/bin/sh

# please edit --datasets_path and --log_directory_path below

python main.py \
--learn_lr True \
--learn_mom True \
--learn_wd True \
--n_lrs 5 \
--n_moms 1 \
--n_wds 1 \
--dataset CIFAR10 \
--n_inner_epochs_per_outer_steps 50 50 50 50 50 50 50 50 50 50 \
--pruning_mode alternate \
--pruning_ratio 0.0 \
--architecture WRN-16-1 \
--init_type xavier \
--init_param 1 \
--init_norm_weights 1 \
--inner_lr_init 0.0 \
--inner_mom_init 0.0 \
--inner_wd_init 0.0 \
--train_batch_size 256 \
--clamp_grads True \
--clamp_grads_range 3 \
--cutout False \
--cutout_length 16 \
--cutout_prob 1.0 \
--val_batch_size 500 \
--val_train_fraction 0.05 \
--val_train_overlap False \
--lr_init_step_size 0.1 \
--mom_init_step_size 0.3 \
--wd_init_step_size 4e-4 \
--lr_step_decay 0.5 \
--mom_step_decay 0.5 \
--wd_step_decay 0.5 \
--clamp_HZ True \
--clamp_HZ_range 100 \
--converged_frac 0.05 \
--retrain_from_scratch True \
--retrain_n_epochs 50 \
--datasets_path /home/me/mydatasets \
--log_directory_path /home/me/mylogs \
--epoch_log_freq 5 \
--outer_step_log_freq 1 \
--seed 1 \
--workers 2 \
--use_gpu True
