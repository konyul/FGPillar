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