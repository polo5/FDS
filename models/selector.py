from models.lenet import *
from models.wresnet import *

def select_model(meta,
                 dataset,
                 architecture,
                 init_type='xavier',
                 init_param=1,
                 device='cpu'):
    """
    Meta models require device to be provided during init.
    """

    if dataset in ['MNIST', 'FashionMNIST']:
        n_classes, n_channels, im_size = 10, 1, 28
        kwargs0 = {'n_classes':n_classes, 'n_channels':n_channels, 'im_size':im_size}
        if architecture == 'LeNet':
            model = MetaLeNet(**kwargs0, device=device) if meta else LeNet(**kwargs0)
        elif architecture == 'LeNet-BN': #debug neg learning rates
            model = MetaLeNetBN(**kwargs0, device=device) if meta else LeNetBN(**kwargs0)
        else:
            raise NotImplementedError

    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        n_channels, im_size = 3, 32
        n_classes = 100 if dataset == 'CIFAR100' else 10
        kwargs0 = {'n_classes':n_classes, 'n_channels':n_channels}
        if architecture == 'LeNet':
            kwargs1 = {'im_size':im_size}
            model = MetaLeNet(**kwargs0, **kwargs1, device=device) if meta else LeNet(**kwargs0, **kwargs1)
        elif architecture == 'LeNetBN':
            kwargs1 = {'im_size':im_size}
            model = MetaLeNetBN(**kwargs0, **kwargs1, device=device) if meta else LeNetBN(**kwargs0, **kwargs1)
        elif architecture == 'WRN-10-1':
            kwargs1 = {'depth':10, 'widen_factor':1, 'dropRate':0.0}
            model = MetaWideResNet(**kwargs0, **kwargs1, device=device) if meta else WideResNet(**kwargs0, **kwargs1)
        elif architecture == 'WRN-16-1':
            kwargs1 = {'depth':16, 'widen_factor':1, 'dropRate':0.0}
            model = MetaWideResNet(**kwargs0, **kwargs1, device=device) if meta else WideResNet(**kwargs0, **kwargs1)
        elif architecture == 'WRN-40-2':
            kwargs1 = {'depth':40, 'widen_factor':2, 'dropRate':0.0}
            model = MetaWideResNet(**kwargs0, **kwargs1, device=device) if meta else WideResNet(**kwargs0, **kwargs1)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    ## Initialization schemes
    if meta:
        model.initialize(init_type=init_type, init_param=init_param, init_norm_weights=1, inplace=True)
    else:
        initialize(model, init_type=init_type, init_param=init_param, init_norm_weights=1)

    return model


if __name__ == '__main__':
    from torchsummary import summary
    from utils.helpers import *

    ## Check meta and normal models do the same calculations
    # x1 = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    # x2 = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    # set_torch_seeds(0)
    # model = select_model(False,  dataset='CIFAR10', architecture='WRN-10-1', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    # set_torch_seeds(0)
    # meta_model = select_model(True,  dataset='CIFAR10', architecture='WRN-10-1', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    # meta_weights = meta_model.get_param()
    #
    # model.train(), meta_model.train() #x1 before and after x2 if comment out eval mode below.
    #
    # model_output = model(x1)
    # meta_model_output = meta_model.forward_with_param(x1, meta_weights)
    # print(float(torch.sum(model_output)), float(torch.sum(meta_model_output)))
    #
    # model_output = model(x2)
    # meta_model_output = meta_model.forward_with_param(x2, meta_weights)
    # print(float(torch.sum(model_output)), float(torch.sum(meta_model_output)))
    #
    # model.eval(), meta_model.eval() #x1 output changes in eval now because running stats were calculated
    # model_output = model(x1)
    # meta_model_output = meta_model.forward_with_param(x1, meta_weights)
    # print(float(torch.sum(model_output)), float(torch.sum(meta_model_output)))


    # x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    #
    # t0 = time.time()
    # model = select_model(True,  dataset='CIFAR10', architecture='WRN-16-1', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    # output = model(x)
    # print("Time taken for forward pass: {} s".format(time.time.time() - t0))
    # print("\nOUTPUT SHAPE: ", output.shape)
    # summary(model, (3, 32, 32), max_depth=5)


    ## Weights init for normal model
    # model = select_model(False,  dataset='CIFAR10', architecture='WRN-16-1', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    # def weights_to_gaussian(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): #TODO separate init for linear layer?
    #         torch.nn.init.normal_(m.weight, mean=0, std=0.1)
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)
    # model.apply(weights_to_gaussian)

    ## Weights init for meta model
    # model = select_model(True,  dataset='CIFAR10', architecture='WRN-16-1', activation='ReLU', norm_type='BN', norm_affine=False, noRes=False)
    # weights = model.get_param()
    # print(len(weights))
    # print(torch.sum(weights))
    # torch.nn.init.normal_(weights, mean=0, std=0.1)
    # print(torch.sum(weights))


    ## Change BN init for meta model
    #model = select_model(False,  dataset='CIFAR10', architecture='LeNet', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    #model = select_model(True,  dataset='CIFAR10', architecture='LeNet', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False)
    #summary(model, (3, 32, 32), max_depth=10)
    # weights = model.get_param()
    # #weights_numels = model._weights_numels
    # #layer_names = model.layer_names()
    # #print(weights.shape[0], sum(weights_numels)) #same
    # #print(len(weights_numels), len(layer_names)) #same
    # gammas_mask, betas_mask = model.get_bn_masks()
    # print(len(weights[gammas_mask]), len(weights[betas_mask]))
    # #print(weights[gammas_mask])
    # print(weights[betas_mask])


    ## Check init
    set_torch_seeds(0)
    model = select_model(False,  dataset='CIFAR10', architecture='ShuffleNetv2-s05', activation='ReLU', norm_type='BN', norm_affine=True, noRes=False,
                         init_type='normal', init_param=1, init_norm_weights=1)

    for n,p in model.named_parameters():
        print(n, float(torch.sum(p)))