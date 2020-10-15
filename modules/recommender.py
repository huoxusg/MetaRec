import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
from collections import defaultdict


def create_module(module_type, **config):
    module_type = module_type.lower()
    if module_type == 'mlp':
        return MLP(**config)
    elif module_type == 'gcn':
        return AttentionGCN(**config)
    elif module_type == 'empty':
        return nn.Sequential()
    else:
        raise NotImplementedError


def activation_method(name):
    """
    :param name: (str)
    :return: torch.nn.Module
    """
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    else:
        return nn.Sequential()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, final_size=0, final_activation="none", normalization="batch_norm",
                 activation='relu'):
        """
        :param input_size:
        :param hidden_layers: [(unit_num, normalization, dropout_rate)]
        :param final_size:
        :param final_activation:
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        fcs = []
        last_size = self.input_size
        for size, to_norm, dropout_rate in hidden_layers:
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            if to_norm:
                if normalization == 'batch_norm':
                    fcs.append(nn.BatchNorm1d(last_size))
                elif normalization == 'layer_norm':
                    fcs.append(nn.LayerNorm(last_size))
            fcs.append(activation_method(activation))
            if dropout_rate > 0.0:
                fcs.append(nn.Dropout(dropout_rate))
        self.fc = nn.Sequential(*fcs)
        if final_size > 0:
            linear = nn.Linear(last_size, final_size)
            linear.bias.data.fill_(0.0)
            finals = [linear, activation_method(final_activation)]
        else:
            finals = []
        self.final_layer = nn.Sequential(*finals)

    def forward(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out


class Recommender(nn.Module):
    def __init__(self, useritem_embeds, user_graph=False, item_graph=False):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.user_graph = user_graph
        self.item_graph = item_graph

    def forward(self, query_users, query_items, with_attr=False):
        if query_users[0].dim() > 1:
            query_users = list(map(lambda x: x.squeeze(0), query_users))
        if query_items[0].dim() > 1:
            query_items = list(map(lambda x: x.squeeze(0), query_items))
        if not with_attr:
            query_users = self.useritem_embeds(*query_users, is_user=True, with_neighbor=self.user_graph)
            query_items = self.useritem_embeds(*query_items, is_user=False, with_neighbor=self.item_graph)
        return query_users, query_items


class InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, mlp_config):
        super(InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = MLP(**mlp_config)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(InteractionRecommender, self).forward(query_users, query_items,
                                                                               with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)
        return self.mlp(query_embeds).squeeze(1)


class EmbedRecommender(Recommender):
    def __init__(self, useritem_embeds, user_config, item_config, user_graph=True, item_graph=True):
        super(EmbedRecommender, self).__init__(useritem_embeds, user_graph, item_graph)
        self.user_model = create_module(**user_config)
        self.item_model = create_module(**item_config)

    def forward(self, query_users, query_items, with_attr=False):
        """
        :param with_attr:
        :param query_users: (batch_size,)
        :param query_items: (batch_size)
        :return:
        """
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users = self.user_model(*query_users)
        query_items = self.item_model(*query_items)
        return (query_users * query_items).sum(dim=1)


class HybridRecommender(Recommender):
    def __init__(self, useritem_embeds, input_size, hidden_layers, final_size, activation='relu',
                 normalization="batch_norm"):
        super(HybridRecommender, self).__init__(useritem_embeds, False, False)
        self.interaction_model = MLP(input_size=2 * input_size, hidden_layers=hidden_layers, activation=activation,
                                     normalization=normalization, final_activation='none', final_size=final_size)
        self.final_layer = nn.Linear(input_size + final_size, 1)

    def forward(self, query_users, query_items, with_attr=False):
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        interactions = torch.cat((query_users, query_items), dim=-1)
        interactions = self.interaction_model(interactions)
        product = query_users * query_items
        concatenation = torch.cat((interactions, product), dim=-1)
        return self.final_layer(concatenation).squeeze(-1)
