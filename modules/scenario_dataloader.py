import random
import torch.utils.data
import copy
from collections import defaultdict


# def filter_kcore(ratings, user_k=5, item_k=5):
#     while True:
#         item_count, user_count = defaultdict(int), defaultdict(int)
#         for user_id, item_id, rating, timestamp in ratings:
#             item_count[item_id] += 1
#             user_count[user_id] += 1
#         user_num, item_num = len(user_count), len(item_count)
#         print(user_num, item_num)
#         # filter the user_set and item_set
#         user_set = set(filter(lambda x: user_count[x] >= user_k, user_count.keys()))
#         item_set = set(filter(lambda x: item_count[x] >= item_k, item_count.keys()))

#         if len(user_set) == user_num and len(item_set) == item_num:
#             break
#         ratings = list(filter(lambda x: x[0] in user_set and x[1] in item_set, ratings))
#     return ratings, user_set, item_set


def batch_generator(data, batch_size, shuffle=True):
    """Yield elements from data in chunks of batch_size."""
    if shuffle:
        sampler = torch.utils.data.RandomSampler(data)
    else:
        sampler = torch.utils.data.SequentialSampler(data)
    minibatch = []
    for idx in sampler:
        minibatch.append(data[idx])
        if len(minibatch) == batch_size:
            yield minibatch
            minibatch = []
    if minibatch:
        yield minibatch


def train_generator(task_ratings, task2candidates, user2itemset, batch_size, few_num=64, negative_ratio=1,
                    shuffle=True):
    user2negatives = []
    # print(len(task2candidates)) ## equal num of train tasks
    # print(len(task_ratings))
    # print(type(task_ratings)) # list
    for idx in range(len(task2candidates)):
        candidates = set(task2candidates[idx])
        user2negatives.append({user_id:list(candidates - itemset) for user_id, itemset in user2itemset[idx].items()})
    while True:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(task_ratings)
        else:
            sampler = torch.utils.data.SequentialSampler(task_ratings)
        for idx in sampler:
            positives = task_ratings[idx]
            candidates = task2candidates[idx]
            random.shuffle(positives)
            if len(positives) > few_num:
                support_pairs, unselected_pairs = positives[:few_num], positives[few_num:]
            else:
                support_pairs, unselected_pairs = positives[:len(positives) // 2], positives[len(positives) // 2:]
                while len(support_pairs) < few_num:
                    support_pairs.append(random.choice(support_pairs))
            if len(unselected_pairs) < batch_size:
                positive_pairs = [random.choice(unselected_pairs) for _ in range(batch_size)]
            else:
                positive_pairs = random.sample(unselected_pairs, batch_size)
            positive_users, positive_items = [pair[0] for pair in positive_pairs], [pair[1] for pair in positive_pairs]
            negative_users = copy.copy(positive_users) * negative_ratio
            negative_items = []
            for i in range(len(negative_users)):
                if len(user2negatives[idx][negative_users[i]]) > 0:
                    negative_item = random.choice(user2negatives[idx][negative_users[i]])
                else:
                    negative_item = random.choice(candidates)
                negative_items.append(negative_item)
            yield support_pairs, candidates, positive_users, positive_items, negative_users, negative_items


def evaluate_generator(task_support, eval_user2itemset, task2candidates, few_num=8):
    def task_iterator(candidates, itemsets):
        for user_id, itemset in itemsets.items():
            if len(itemset) > 0:
                positive_users, positive_items = [], []
                for item_id in itemset:
                    positive_users.append(user_id)
                    positive_items.append(item_id)
                negative_users, negative_items = [], []
                for item_id in candidates:
                    if item_id not in itemset:
                        negative_users.append(user_id)
                        negative_items.append(item_id)
                if len(positive_users) > 0 and len(negative_users) > 0:
                    yield positive_users, positive_items, negative_users, negative_items

    for idx in range(len(eval_user2itemset)):
        if few_num is None:
            few_size = len(task_support[idx])
        else:
            few_size = few_num
        # Consistent for different iterations
        support_pairs = task_support[idx][:few_size]
        yield support_pairs, task2candidates[idx], task_iterator(task2candidates[idx], eval_user2itemset[idx])
