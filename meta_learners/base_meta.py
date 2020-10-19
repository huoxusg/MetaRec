import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math, itertools
from collections import defaultdict
# from modules.recommender import MLP


class StopControl(nn.Module):
        def __init__(self, input_size, hidden_size):
            nn.Module.__init__(self)
            self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1)
            self.output_layer.bias.data.fill_(0.0)
            self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
            self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

        def forward(self, inputs, hx):
            if hx is None:
                hx = (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))
            h, c = self.lstm(inputs, hx)
            return torch.sigmoid(self.output_layer(h).squeeze()), (h, c)


class GradientModel(nn.Module):
    def __init__(self, user_neighbors, item_neighbors, useritem_embeds, model, loss_function, attack_model=None,
                 step=10, min_step=None, flexible_step=False, hidden_input=True, addition_params=None,
                 batch_size=64, learn_bias=False,
                 user_graph=True, item_graph=False):
        nn.Module.__init__(self)
        self.flexible_step = flexible_step
        if min_step is None:
            min_step = step
        self.min_step = min_step
        if addition_params is None:
            addition_params = []
        self.user_neighbors = user_neighbors
        self.item_neighbors = item_neighbors
        self.useritem_embeds = useritem_embeds
        self.model = model
        self.learned_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # set meta params by adding 'share_'
                # print(name, module)
                # mlp.fc.0 Linear(in_features=128, out_features=64, bias=True)
                # mlp.fc.2 Linear(in_features=64, out_features=32, bias=True)
                # mlp.fc.4 Linear(in_features=32, out_features=16, bias=True)
                # mlp.final_layer.0 Linear(in_features=16, out_features=1, bias=True)
                self.learned_params.append((name, 'weight'))
                self.init_learned_param(module, 'weight')
                if learn_bias and module.bias is not None:
                    self.learned_params.append((name, 'bias'))
                    self.init_learned_param(module, 'bias')
        named_modules = dict(self.model.named_modules())
        # for layer_name, param_name in addition_params:
        #     if not (layer_name, param_name) in self.learned_params:
        #         self.learned_params.append((layer_name, param_name))
        #         module = named_modules[layer_name]
        #         self.init_learned_param(module, param_name)
        self.loss_function = loss_function
        # gradient, parameter, last state, loss
        self.max_step = step
        self.batch_size = batch_size
        self.user_graph = user_graph
        self.item_graph = item_graph
        self.attack_model = attack_model
        self.aim_parameters = []
        if self.flexible_step:
            self.hidden_input = hidden_input
            if hidden_input:
                stop_input_size = len(self.learned_params) * lstm_config['hidden_size']
            else:
                stop_input_size = len(self.learned_params) + 1
            hidden_size = stop_input_size * 2
            self.stop_gate = StopControl(stop_input_size, hidden_size)

    @staticmethod
    def init_learned_param(module, param_name):
        param = getattr(module, param_name)
        del module._parameters[param_name]
        module.register_parameter('share_' + param_name, param)
        # Gradients propagating to the cloned tensor will propagate to the original tensor
        setattr(module, param_name, param.data.clone())

    def init(self):
        # self.current_step = 0
        self.store = {
            'loss': [],
            'grads': [],
            'stop_gates': []
        }
        self.aim_parameters = []
        named_modules = dict(self.model.named_modules())
        for layer_name, param_name in self.learned_params:
            layer = named_modules[layer_name]
            setattr(layer, param_name, getattr(layer, 'share_' + param_name).clone().requires_grad_(True))
            self.aim_parameters.append(getattr(layer, param_name))
        if self.flexible_step:
            self.stop_hx = None

    def update(self, parameters, loss, grads):
        raise NotImplementedError

    def stop(self, step, loss, grads):
        if self.flexible_step:
            if step < self.max_step:
                if self.hidden_input:
                    hxs = list(map(lambda x: x[0].mean(dim=0), self.hxs))
                    # 1, param_num * hidden_size
                    inputs = torch.cat(hxs, dim=0).unsqueeze(0)
                else:
                    grad_norms = list(map(lambda x: x.detach().norm(), grads))
                    inputs = grad_norms + [loss.detach()]
                    inputs = torch.stack(inputs, dim=0).unsqueeze(0)
                    inputs = self.smooth(inputs)[0]
                stop_gate, self.stop_hx = self.stop_gate(inputs, self.stop_hx)
                return stop_gate
        return loss.new_zeros(1, dtype=torch.float)

    def forward(self, *input, **kwargs):
        for _ in self.step_forward(*input, **kwargs):
            pass

    @staticmethod
    def smooth(weight, p=10, eps=1e-20):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        return log_weight, sign

    def step_forward(self, support_pairs, candidates, step_forward=False):
        """
        :param query_items: (1, batch_size,)
        :param query_users: (1, batch_size,)
        :param support_pairs: (1, few_size, 2)
        :param candidates: list of python
        :return:
        """
        named_modules = dict(self.model.named_modules())
        support_users, support_items = support_pairs[:, 0], support_pairs[:, 1]
        self.device = support_users.device
        support_users = self.useritem_embeds(*self.user_neighbors(support_users), is_user=True)
        support_items = self.useritem_embeds(*self.item_neighbors(support_items), is_user=False)
        for step in range(self.max_step):
            if len(support_pairs) >= self.batch_size:
                rand_index = random.sample(range(len(support_pairs)), self.batch_size)
            else:
                rand_index = [random.randrange(len(support_pairs)) for _ in range(self.batch_size)]
            selected_users, positive_items = list(map(lambda x: x[rand_index], support_users)), list(
                map(lambda x: x[rand_index], support_items))
            negative_items = []
            for idx in rand_index:
                negative_item = random.choice(candidates)
                while negative_item == support_pairs[idx, 1]:
                    negative_item = random.choice(candidates)
                negative_items.append(negative_item)
            negative_items = torch.tensor(negative_items, dtype=torch.long, device=self.device)
            negative_items = self.useritem_embeds(*self.item_neighbors(negative_items), is_user=False,
                                                  with_neighbor=self.item_graph)
            # if self.attack_model is not None:
            #     attack_users, attack_positives, attack_negatives = self.attack_model(self.model, selected_users,
            #                                                                          positive_items, negative_items)
            #     selected_users = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
            #                               zip(selected_users, attack_users)))
            #     positive_items = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
            #                               zip(positive_items, attack_positives)))
            #     negative_items = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
            #                               zip(negative_items, attack_negatives)))
            positive_values = self.model(selected_users, positive_items, with_attr=True)
            negative_values = self.model(selected_users, negative_items, with_attr=True)
            loss = self.loss_function(positive_values, negative_values)
            grads = torch.autograd.grad(loss, self.aim_parameters)
            self.store['loss'].append(loss.item())
            self.store['grads'].append(list(map(lambda x: x.norm().item(), grads)))
            # bug: self.current_step -> step
            stop_gate = self.stop(step, loss, grads)
            self.store['stop_gates'].append(stop_gate.item())
            if step >= self.min_step and random.random() < stop_gate:
                break
            if step_forward:
                yield stop_gate
            self.aim_parameters = self.update(self.aim_parameters, loss, grads)
            for weight, (layer_name, param_name) in zip(self.aim_parameters, self.learned_params):
                layer = named_modules[layer_name]
                setattr(layer, param_name, weight)
        # if query_users is not None and query_items is not None:
        #     if not with_attr:
        #         query_users, query_items = query_users.squeeze(0), query_items.squeeze(0)
        #         query_users = self.user_neighbors(query_users)
        #         query_items = self.item_neighbors(query_items)
        #     return self.model(query_users, query_items, with_attr=with_attr)
