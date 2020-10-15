import random, os
import torch
import torch.nn as nn
import numpy as np
import json

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "npy":
        return np.load(path)
    elif suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def divide_dataset(dataset, valid_ratio=0.1, test_ratio=0.1):
    train_data, valid_data, test_data = [], [], []
    index = list(range(len(dataset)))
    random.shuffle(index)
    valid_size, test_size = round(len(dataset) * valid_ratio), round(len(dataset) * test_ratio)
    valid_index, test_index = set(index[:valid_size]), set(index[valid_size: valid_size + test_size])
    for i, data in enumerate(dataset):
        if i in valid_index:
            valid_data.append(data)
        elif i in test_index:
            test_data.append(data)
        else:
            train_data.append(data)
    return train_data, valid_data, test_data


def task_preprocess(tasks):
    '''
    support, candidates (all items), truth
    '''
    task2candidates = [set(map(lambda x: x[1], ratings)) for ratings in tasks] # rating: (user, item)
    task2candidates = [list(candidates) for candidates in task2candidates]
    user2itemset = []
    for ratings in tasks:
        itemset = {}
        for user_id, item_id in ratings:
            if user_id not in itemset:
                itemset[user_id] = set()
            itemset[user_id].add(item_id)
        user2itemset.append(itemset)
    return tasks, task2candidates, user2itemset


# divide the support and evaluate data
'''
对每个task，按user分类，随机切分support和eval
'''
def divide_support(task_ratings, support_limit=512, evaluate_limit=None):
    task_usertruth = []
    for ratings in task_ratings:
        task_usertruth.append({})
        for user_id, *others in ratings:
            if user_id not in task_usertruth[-1]:
                task_usertruth[-1][user_id] = []
            if others not in task_usertruth[-1][user_id]:
                task_usertruth[-1][user_id].append(others)
    divide_data = []
    print(len(task_usertruth))
    for i in range(len(task_usertruth)):
        # key = task_ratings[i][0]
        support_ratings, eval_ratings = [], []
        u2i = list(task_usertruth[i].items())
        random.shuffle(u2i)
        # print(len(u2i))
        for j, (user_id, itemset) in enumerate(u2i):
            if j < len(u2i) // 2 and len(support_ratings) < support_limit:
                aim = support_ratings
            elif evaluate_limit is None or len(eval_ratings) < evaluate_limit:
                aim = eval_ratings
            else:
                break
            for item_id in itemset:
                aim.append((user_id, *item_id))
        # print(len(support_ratings))
        # print(len(eval_ratings))
        # print('******')
        divide_data.append((support_ratings, eval_ratings))
    return divide_data


def filter_statedict(module):
    state_dict = module.state_dict(keep_vars=True)
    non_params = []
    for key, value in state_dict.items():
        if not value.requires_grad:
            non_params.append(key)
    state_dict = module.state_dict()
    for key in non_params:
        del state_dict[key]
    return state_dict


class UserItemEmbeds(nn.Module):
    def __init__(self, user_embeds, item_embeds):
        nn.Module.__init__(self)
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds

    def forward(self, nodes, neighbors=None, degrees=None, is_user=True, with_neighbor=True):
        if is_user:
            if with_neighbor and neighbors is not None and degrees is not None:
                return self.user_embeds(nodes), self.item_embeds(neighbors), degrees
            else:
                return (self.user_embeds(nodes),)
        else:
            if with_neighbor and neighbors is not None and degrees is not None:
                return self.item_embeds(nodes), self.user_embeds(neighbors), degrees
            else:
                return (self.item_embeds(nodes),)


class NeighborDict(nn.Module):
    def __init__(self, neighbor_dict=None, max_degree=512, padding_idx=0):
        nn.Module.__init__(self)
        self.neighbor_dict = neighbor_dict
        self.max_degree = max_degree
        self.flag = nn.Parameter(torch.empty(0), requires_grad=False)
        self.padding_idx = padding_idx

    def forward(self, nodes):
        # print(torch.is_tensor(nodes))
        # if not torch.is_tensor(nodes):
        #     nodes = torch.tensor(nodes, dtype=torch.long, device=self.flag.device)
        return (nodes,)

        # if torch.is_tensor(nodes):
        #     if self.neighbor_dict is not None:
        #         neighbors = [random.sample(self.neighbor_dict[idx.item()], self.max_degree) if len(
        #             self.neighbor_dict[idx.item()]) > self.max_degree else self.neighbor_dict[idx.item()] for idx in
        #                      nodes]
        # else:
        #     if self.neighbor_dict is not None:
        #         neighbors = [random.sample(self.neighbor_dict[idx], self.max_degree) if len(
        #             self.neighbor_dict[idx]) > self.max_degree else self.neighbor_dict[idx] for idx in nodes]
        #     nodes = torch.tensor(nodes, dtype=torch.long, device=self.flag.device)
        # if self.neighbor_dict is not None:
        #     degrees = torch.tensor(list(map(len, neighbors)), dtype=torch.long, device=self.flag.device)
        #     neighbors = list2tensor(neighbors, self.padding_idx, device=self.flag.device)
        #     return nodes, neighbors, degrees
        # else:
        #     return (nodes,)