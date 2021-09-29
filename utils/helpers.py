import csv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import shutil
import datetime
import json
import os
import argparse
import gc

import numpy as np
import torchvision
import functools
import time
import warnings
#warnings.simplefilter("ignore", UserWarning)


### Metrics


class AggregateTensor(object):
    """
    Computes and stores the average of stream.
    Mostly used to average losses and accuracies.
    Works for both scalars and vectors but input needs
    to be a pytorch tensor.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0001  # DIV/0!
        self.sum = 0
        #self.sum2 = 0

    def update(self, val, w=1):
        """
        :param val: new running value
        :param w: weight, e.g batch size
        Turn everything into floats so that we don't keep bits of the graph
        """
        self.sum += w * val.detach()
        self.count += w


    def avg(self):
        return self.sum / self.count

    # def std(self):
    #     return np.sqrt(self.sum2/self.count - self.avg()**2)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res

def avg_entropy(pmf):
    """
    :param pmf: pytorch tensor pmf of shape [batch_size, n_classes]
    :return: average entropy of pmf across entire batch
    """
    #assert
    assert ((pmf>=0)*(pmf<=1.00001)).all(), "All inputs must be in range [0,1] but min/max is {}/{}".format(float(torch.min(pmf)), float(torch.max(pmf)))
    p_log_p = torch.log2(torch.clamp(pmf, min=0.0001, max=1.0))*pmf #log(0) causes error
    return torch.mean(-p_log_p.sum(1))

def avg_max(pmf):
    """
    :param pmf: pytorch tensor pmf of shape [batch_size, n_classes]
    when learned the pmf doesn't have to be within [0,1]
    :return: average of max predictions of pmf across entire batch
    """
    assert ((pmf >= 0) * (pmf <= 1)).all(), "All inputs must be in range [0,1]"
    return torch.mean(torch.max(pmf, 1)[0])

def onehot(targets, n_classes):
    """
    Convert labels of form [[2], [7], ...] to
    [0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,0], ...]
    :param targets:
    :param n_classes:
    :param device:
    :return:
    """
    return torch.zeros((targets.shape[0], n_classes), device=targets.device).scatter(1, targets.unsqueeze(-1), 1)

def gc_tensor_view(verbose=True):
    """
    Doesn't catch intermediate variables stored by Pytorch graph
    if they are not in the Python scope.
    assumes all tensors are torch.float() i.e. 32 bit (4MB)
    """
    total_MB_size = 0
    object_counts = {}
    object_MBs = {}
    if verbose: print('\n------- TENSORS SEEN BY GARBAGE COLLECTOR -------')

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                MB_size = np.prod(obj.size()) * 4 / 1024**2 #assume float32
                total_MB_size += MB_size #str(type(obj))
                key = str(obj.size())[10:]
                object_counts[key] = object_counts.get(key, 0) + 1
                object_MBs[key] = MB_size
        except:
            pass

    if verbose:
        object_totals = {k:object_counts[k] * object_MBs[k] for k in object_MBs.keys()}
        for key, value in sorted(object_totals.items(), key=lambda item: item[1], reverse=True):
            print("{}  x  {}  ({:.0f}MB) = {:.0f}MB".format(object_counts[key], key,  object_MBs[key], object_counts[key]*object_MBs[key]))


    print("TOTAL MEMORY USED BY PYTORCH TENSORS: {:.0f} MB".format(total_MB_size))

def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        #print('\n------------------------------')
        print(f"--- Ran func {func.__name__!r} in {format_time(run_time)} ---")
        #print('------------------------------\n')

        return value
    return wrapper_timer

### Data view and read

def unormalize_CIFAR10_image(image):
    return image*torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)

# def plot_image(input, unormalize=False):
#     if len(input.shape) > 3:
#         print("Use plot_images function instead!")
#         raise NotImplementedError
#     npimg = input.numpy()
#     if unormalize:
#         npimg = npimg * np.array([0.2023, 0.1994, 0.2010]).reshape(3,1,1) + np.array([0.4914, 0.4822, 0.4465]).reshape(3,1,1)
#     npimg = np.transpose(npimg, (1, 2, 0))
#     if npimg.shape[-1] != 3:
#         npimg = npimg[:, :, 0]
#     #print(npimg.shape)
#
#     fig = plt.figure(figsize=(20, 20))
#     ax = fig.add_subplot(111)
#     ax.axis('off')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#
#     ax.imshow(npimg, cmap='gray')
#     plt.show()
#     return fig

# def plot_images(batch, padding=2, unormalize=False):
#     if len(batch.shape) == 3:
#         plot_image(batch, unormalize=unormalize)
#     elif len(batch.shape) == 4:
#         n_images = batch.shape[0]
#         if n_images == 1:
#             plot_image(batch[0], unormalize=unormalize)
#         else:
#             grid_img = torchvision.utils.make_grid(batch, nrow=int(np.ceil(np.sqrt(n_images))), padding=padding)
#             plot_image(grid_img, unormalize=unormalize)

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def delete_files_from_name(folder_path, file_name, type='contains'):

    assert type in ['is', 'contains']
    for f in os.listdir(folder_path):
        if (type=='is' and file_name==f) or (type=='contains' and file_name in f):
            os.remove(os.path.join(folder_path, f))

def copy_file(file_path, folder_path):
    destination_path = os.path.join(folder_path, os.path.basename(file_path))
    shutil.copyfile(file_path, destination_path)

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%dh%02dm%02ds" % (hours, minutes, seconds)

def create_empty_file(path):
    """Easy way to log final test accuracy in some experiment folder"""
    with open(path, 'w+') as f: f.write("NA")

if __name__ == '__main__':

    from time import time
    import torch

    ## Test AggregateTensor
    # x = np.random.rand(1000)*50
    # w = np.random.rand(1000)*5
    # true_mu = w@x/np.sum(w)
    # true_std = np.sqrt(np.sum(w*(x-true_mu)**2)/((len(x)-1)*np.sum(w)/len(x)))
    #
    # t0 = time.time()
    # a = "yolo"
    # print("Init of string takes: {} us".format(1e6*(time()-t0)))
    #
    # t0 = time.time()
    # meter = AggregateTensor()
    # print("Init of AggregateTensor takes: {} us".format(1e6*(time()-t0)))
    #
    # t0 = time.time()
    # a = 1000*5
    # print("Multiplication takes: {} us".format(1e6 * (time.time() - t0)))
    #
    # t = 0
    # for val,weight in zip(x,w):
    #     t0 = time.time()
    #     meter.update(val, weight)
    #     t += time.time() - t0
    # print("Avg update time: {} us".format(1e6*t/len(x)))
    #
    # print(true_mu, meter.avg())
    # #print(np.std(x), true_std, meter.std())
    #
    # ### Test AggregateDict
    # keys = ['loss', 'acc', 'yolo']
    # meter = AggregateDict()
    #
    # values = [[1,2,3], [3,4,5], [1,1,1]]
    # true_mus = [np.mean(el) for el in values]
    # true_stds = [np.std(el) for el in values]
    #
    # for i in range(3):
    #     dict = {k: v[i] for k,v in zip(keys, values)}
    #     print(dict)
    #     meter.update(val=dict, w=1)
    #
    #
    # print(true_mus, meter.avg())
    #print(true_stds, meter.std())

    ### Test cutout

    # Data loader tests
    # from time import time
    # import torchvision.datasets as datasets
    # import torchvision.transforms as transforms
    #
    # device = torch.device('cpu')
    # dataset_path = "~/Datasets/Pytorch/"
    #
    #
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     Cutout(n_holes=1, length=16, cutout_proba=0.5)])
    #
    # dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform)
    # loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=5,
    #     shuffle=False, drop_last=False, num_workers=4)
    #
    # for x,y in loader:
    #     print(x.shape, y.shape)
    #     image = x[4]#*torch.Tensor[0.2023, 0.1994, 0.2010])-torch.Tensor([0.4914, 0.4822, 0.4465]
    #     plot_image(image)
    #     break


    ### Test entropy
    # output = torch.Tensor([[0.1, 0.5, 0.4],
    #                        [0.3,0.3,0.4],
    #                        [0.99, 0.005, 0.005],
    #                        [0.5, 0.5, 0.000001]])
    #
    # print(avg_entropy(output))

    ## Test Dataloader indices
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset = datasets.MNIST("~/Datasets/Pytorch/", train=True, download=True, transform=transform)
    # dataset = DatasetWithIndices(dataset)
    # loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=5,shuffle=True,drop_last=True,num_workers=1)
    #
    # cnt = 0
    # for x, y, indices in loader:
    #     print(x.shape, y.shape, indices.shape)
    #     print(indices)
    #     if cnt>5:
    #         break
    #     cnt+=1

    #print(len(dataset), len(loader))


    ## Test Dataloader coefficient
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset = datasets.MNIST("~/Datasets/Pytorch/", train=True, download=True, transform=transform)
    # dataset = DatasetWithLearnableCoefficients(dataset)
    # loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=5,shuffle=True,drop_last=True,num_workers=1)
    #
    # cnt = 0
    # for x, y, indices in loader:
    #     print(x.shape, y.shape, indices.shape)
    #     print(indices)
    #     if cnt>5:
    #         break
    #     cnt+=1
    #
    # print(len(dataset), len(loader))


    ## Test Corrupter
    # corrupter = Corrupter(n_images=50000, fraction_to_corrupt=0.1, n_classes=10)
    # indices = torch.arange(10000, 10000+260, dtype=torch.long)
    # targets = torch.arange(10, dtype=torch.long).repeat(26)
    #
    # t0 = time.time()
    # corrupted = corrupter(indices, targets)
    # print(time.time() - t0)
    #
    # print(corrupted)
    # print(len(corrupted))


    ## Test Aggregate for vector
    #a = torch.tensor([1,2,3])
    #b = torch.tensor([3,4,5])

    a = torch.FloatTensor([1])
    b = torch.FloatTensor([2])

    c = AggregateVector()
    c.update(a)
    c.update(b)
    print(c.avg())


    ###
    pass




