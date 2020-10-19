import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math, itertools
from collections import defaultdict
# from modules.recommender import MLP
from meta_learners.base_meta import GradientModel


class MetaLSTM(nn.Module):
    def __init__(self, hidden_size, layer_norm=False, input_gate=True, forget_gate=True):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        # gradient(2), param(2), loss
        self.lstm = nn.LSTMCell(input_size=5, hidden_size=hidden_size)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        if self.input_gate:
            self.lr_layer = nn.Linear(hidden_size, 1)
            self.lrs = []
        else:
            self.output_layer = nn.Linear(hidden_size, 1)
            self.dets = []
        if self.forget_gate:
            self.fg_layer = nn.Linear(hidden_size, 1)
            self.fgs = []
        self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
        self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

    def weight_init(self):
        if self.input_gate:
            nn.init.xavier_normal_(self.lr_layer.weight)
            self.lr_layer.bias.data.fill_(0.0)
        else:
            nn.init.xavier_normal_(self.output_layer.weight)
            self.output_layer.weight.data /= 1000.0
            self.output_layer.bias.data.fill_(0.0)
        if self.forget_gate:
            nn.init.xavier_normal_(self.fg_layer.weight)
            self.fg_layer.bias.data.fill_(5.0)
        self.lstm.reset_parameters()
        hidden_size = self.lstm.hidden_size
        self.lstm.bias_ih.data[hidden_size // 4: hidden_size // 2].fill_(1.0)
        self.lstm.bias_hh.data[hidden_size // 4: hidden_size // 2].fill_(0.0)
        self.h_0.data = torch.randn((self.hidden_size,), requires_grad=True)
        self.c_0.data = torch.randn((self.hidden_size,), requires_grad=True)

    def forward(self, grad_norm, grad_sign, param_norm, param_sign, loss_norm, hx):
        batch_size = grad_norm.size(0)
        inputs = torch.stack((grad_norm, grad_sign, param_norm, param_sign, loss_norm.expand(grad_norm.size(0))),
                                dim=1)
        if hx is None:
            self.lrs = []
            if self.forget_gate:
                self.fgs = []
            hx = (self.h_0.expand((batch_size, -1)), self.c_0.expand((batch_size, -1)))
        h, c = self.lstm(inputs, hx)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        if self.input_gate:
            lr = torch.sigmoid(self.lr_layer(h))
        else:
            lr = self.output_layer(h)
        self.lrs.append(lr.mean().item())
        if self.forget_gate:
            fg = torch.sigmoid(self.fg_layer(h))
            self.fgs.append(fg.mean().item())
            return lr, fg, (h, c)
        else:
            return lr, (h, c)


class LSTMLearner(GradientModel):
    def __init__(self, *params, input_gate, forget_gate, lstm_config, **meta_config):
        super(LSTMLearner, self).__init__(*params, **meta_config)
        meta_lstms = []
        for _ in self.learned_params:
            lstm = MetaLSTM(input_gate=input_gate, forget_gate=forget_gate, **lstm_config)
            lstm.weight_init()
            meta_lstms.append(lstm)
        # print(params)
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.hxs = []
        self.meta_lstms = nn.ModuleList(meta_lstms)

    def init(self):
        GradientModel.init(self)
        self.store['input_gates'] = []
        self.store['forget_gates'] = []
        self.hxs = []
        for _ in self.learned_params:
            self.hxs.append(None)

    def update(self, parameters, loss, grads):
        loss = loss.detach()
        smooth_loss = self.smooth(loss)[0]
        new_parameters = []
        lrs, fgs = [], []
        for num, (meta_lstm, param, grad, hx) in enumerate(
                zip(self.meta_lstms, parameters, grads, self.hxs)):
            grad.clamp_(-1.0, 1.0)
            flat_grad, flat_param = grad.view(-1), param.detach().view(-1)
            smooth_grad, smooth_param = self.smooth(flat_grad), self.smooth(flat_param)
            if self.forget_gate:
                lr, fg, hx = meta_lstm(*smooth_grad, *smooth_param, smooth_loss, hx)
                lrs.append(lr.mean().item())
                fgs.append(fg.mean().item())
                lr, fg = lr.view_as(grad), fg.view_as(grad)
                weight = fg * param
            else:
                lr, hx = meta_lstm(*smooth_grad, *smooth_param, smooth_loss, hx)
                lr = lr.view_as(grad)
                weight = param
            if self.input_gate:
                weight -= lr * grad
            else:
                weight += lr
            new_parameters.append(weight)
            self.hxs[num] = hx
        self.store['input_gates'].append(lrs)
        self.store['forget_gates'].append(fgs)
        return new_parameters
