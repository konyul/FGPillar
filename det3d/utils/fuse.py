import torch
import copy
import torch.nn as nn

def bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    
    w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1)
    b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False

def fold_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fold_conv_bn_eval_sequential(sequential):
    conv = sequential[0]
    bn = sequential[1]
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    
    out = nn.Sequential(fused_conv, nn.ReLU())
    return out

def fold_conv_bn_eval_sequentialv2(sequential):
    out = []
    for i in range(2):
        conv = sequential[i*3]
        bn = sequential[i*3+1]
        fused_conv = copy.deepcopy(conv)
        fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                                bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        out.append(fused_conv)
        out.append(nn.ReLU())
    out = nn.Sequential(*out)
    return out

def fold_conv_bn2_eval_sequential(sequential):
    out=[]
    conv = sequential[0]
    bn = sequential[1]
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    out += [fused_conv,nn.ReLU()]
    conv = sequential[3]
    bn = sequential[4]
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    out += [fused_conv,nn.ReLU()]
    out = nn.Sequential(*out)
    return out