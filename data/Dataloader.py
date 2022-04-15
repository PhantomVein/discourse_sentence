from data.Explain import Explain
from torch.nn.utils.rnn import pad_sequence
import re
import collections
import numpy as np
import torch
from itertools import groupby
import random
import xml.etree.ElementTree as element_tree


def read_slice_hotel_corpus(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        context = f.read()
        count = 1
        name = []
        item = []
        score = []
        comment = []
        polar = []
        for sentence in context.split('\n'):
            if count == 1:
                name.append(sentence)
            elif count == 2:
                item.append(sentence)
            elif count == 3:
                score.append(sentence)
            elif count == 4:
                comment.append(sentence)
            elif count == 5:
                polar.append(sentence)
            count = (count + 1) % 6

    for explain in explain_in_opinion(comment):
        data.append(explain)

    data=list(set(data)) 
    total_num = len(data)
    train_data, dev_data, test_data  = data[:total_num*6//10], data[total_num*6//10:total_num*8//10], data[total_num*8//10:]
    print('train data:',len(train_data))
    print('dev data:',len(dev_data))
    print('test data:',len(test_data))

    length_count = []
    class_label = collections.defaultdict(int)
    for i in data:
        length_count.append(len(i.context))
        class_label[i.is_explain] += 1
    print('label nums:', class_label)
    interval = 40
    for k,g in groupby(sorted(length_count),key=lambda x:x//interval):
        print('{}-{}:{}'.format(k*interval,(k+1)*interval-1,len(list(g))))

    return train_data, dev_data, test_data

def read_slice_phone_corpus(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        context = f.read()
        product = []
        score = []
        comment = []
        polar = []
        for one_user in context.split('\n\n'):
            information_list = one_user.split('\n')
            if len(information_list) > 3:
                product.append(information_list[0])
                score.append(information_list[1])
                comment.extend(information_list[2:-1])
                polar.append(information_list[-1])

    for explain in explain_in_opinion(comment):
        data.append(explain)

    data=list(set(data)) 
    total_num = len(data)
    train_data, dev_data, test_data  = data[:total_num*6//10], data[total_num*6//10:total_num*8//10], data[total_num*8//10:]
    print('train data:',len(train_data))
    print('dev data:',len(dev_data))
    print('test data:',len(test_data))

    length_count = []
    class_label = collections.defaultdict(int)
    for i in data:
        length_count.append(len(i.context))
        class_label[i.is_explain] += 1
    print('label nums:', class_label)
    interval = 40
    for k,g in groupby(sorted(length_count),key=lambda x:x//interval):
        print('{}-{}:{}'.format(k*interval,(k+1)*interval-1,len(list(g))))

    return train_data, dev_data, test_data


def explain_in_opinion(comment):
    for statement in comment:
        pattern_fac = re.compile(r'<exp-fac.*?>(.*?)</exp-fac.*?>')
        pattern_rea = re.compile(r'<exp-rea.*?>(.*?)</exp-rea.*?>')
        pattern_con = re.compile(r'<exp-con.*?>(.*?)</exp-con.*?>')
        pattern_sug = re.compile(r'<exp-sug.*?>(.*?)</exp-sug.*?>')
        factor = pattern_fac.findall(statement)
        reality = pattern_rea.findall(statement)
        condition = pattern_con.findall(statement)
        suggestion = pattern_sug.findall(statement)
        span = factor + reality + condition + suggestion
        if span:
            yield Explain(True, span, statement)
        else:
            yield Explain(False, None, statement)



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def inst(data):
    return data


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    batch_gold_label = []
    for explain in batch:
        batch_gold_label.append(vocab.label2id(explain.is_explain))
    batch_gold_label = torch.LongTensor(batch_gold_label)
    return batch, batch_gold_label


def batch_pretrain_variable(batch, vocab):
    length = len(batch[0].tokens)
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b].tokens) > length: length = len(batch[b].tokens)

    bert_ids = torch.zeros((batch_size, length), dtype=torch.long)
    masks = torch.zeros_like(bert_ids)
    lengths = []

    b = 0
    for review in batch:
        tokens = vocab.tokenize(review.context)
        ids = vocab.convert_tokens_to_ids(tokens)
        lengths.append(len(tokens))
        for i in range(len(tokens)):
            bert_ids[b, i] = ids[i]
            masks[b, i] = 1
        b += 1
    return bert_ids, lengths, masks

def batch_discourse_variable(batch, vocab):
    edu_masks = []
    batch_edu_explanatory = []
    batch_edge = []
    batch_edge_type = []
    edu_lengths = []
    sentence_max_length = len(batch[0].tokens)
    edu_max_length = 0
    batch_size = len(batch)
    relation2type = {'解说类':0, '转折类':1, '因果类':2, '并列类':3}
    batch_is_main = []
    for b in range(0, batch_size):
        if len(batch[b].tokens) > sentence_max_length: sentence_max_length = len(batch[b].tokens)
        edu_mask, edu_explanatory, edge, edge_relation, is_main_edge = discourse_tree(batch[b], vocab)
        edu_len = len(edu_explanatory)
        if edu_len > edu_max_length: edu_max_length = edu_len
        edu_lengths.append(edu_len)
        edu_masks.append(edu_mask)
        batch_edu_explanatory.append(edu_explanatory)
        batch_edge.append(edge)
        for relation, is_main in zip(edge_relation, is_main_edge):
            batch_edge_type.append(relation2type[relation])
            batch_is_main.append(is_main)
    
    batch_edu_mask = torch.zeros((batch_size, edu_max_length, sentence_max_length), dtype=torch.long, requires_grad=False)
    for b in range(0, batch_size):
        batch_edu_mask[b, 0:edu_masks[b].size()[0], 0:edu_masks[b].size()[1]] = edu_masks[b]
        batch_edge[b] = batch_edge[b] + edu_max_length * b
    batch_edu_explanatory = pad_sequence(batch_edu_explanatory, batch_first=True, padding_value=0)
    batch_edge = torch.cat(batch_edge, 0).T
    batch_edge_type = torch.tensor(batch_edge_type, dtype=torch.float, requires_grad=False).unsqueeze(1)
    batch_is_main = torch.tensor(batch_is_main, dtype=torch.float, requires_grad=False).unsqueeze(1)
    edu_lengths = torch.tensor(edu_lengths, dtype=torch.long, requires_grad=False)
    return batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main

# 静态变量hack
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(function_map = {})
def discourse_tree(review, vocab):
    if review in discourse_tree.function_map:
        return discourse_tree.function_map[review]
    node = []
    edge = []
    edge_relation = []
    is_main_edge = []
    tokens = vocab.tokenize(review.context)
    explain_masks = torch.zeros(len(tokens), dtype=torch.long, requires_grad=False)
    if review.spans:
        for span in review.spans:
            span_token = vocab.tokenize(span)
            start = string_match(tokens,span_token)
            explain_masks[start:start+len(span_token)] = 1
        
    trans_xml = eval(review.discourse).encode('utf-8','replace').decode('utf-8','replace')

    root = element_tree.fromstring(trans_xml)
    i = 0
    stack = []
    for child in root.findall(".//EDU"):
        node.append((i, child[0].text))
        i += 1
    
    type2relation = ['解说类', '转折类', '因果类', '并列类'] 
    edu_num = len(node)
    for edu_i in range(edu_num):
        # reduce
        while len(stack) >= 2 and random.random()>0.5:
            right_i, right_edu = stack.pop()
            left_i, left_edu = stack.pop()
            stack.append((i, left_edu + right_edu))
            
            node.append(stack[-1])
            edge.append([right_i, i])
            edge_type = random.randint(0,3) 
            edge_relation.append(type2relation[edge_type])
            edge.append([left_i, i])
            edge_relation.append(type2relation[edge_type])
            is_main_edge.extend([random.randint(0,1) , random.randint(0,1) ])
            i += 1
        # shift
        stack.append(node[edu_i])
    # reduce
    while len(stack) >= 2:
        right_i, right_edu = stack.pop()
        left_i, left_edu = stack.pop()
        stack.append((i, left_edu + right_edu))
        
        node.append(stack[-1])
        edge.append([right_i, i])
        edge_type = random.randint(0,3) 
        edge_relation.append(type2relation[edge_type])
        edge.append([left_i, i])
        edge_relation.append(type2relation[edge_type])
        is_main_edge.extend([random.randint(0,1) , random.randint(0,1) ])
        i += 1
        

    content = node[-1][1]
    edu_masks = torch.zeros((len(node), len(tokens)), dtype=torch.long, requires_grad=False)
    tokens = vocab.tokenize(content)
    for i, element in node:
        element_token = vocab.tokenize(element)
        start = string_match(tokens,element_token)
        edu_masks[i, start:start+len(element_token)] = 1
    edu_explanatory = torch.matmul(edu_masks, explain_masks.t()).bool().long()
    edge = torch.tensor(edge, dtype=torch.long, requires_grad=False)
    discourse_tree.function_map[review] = (edu_masks, edu_explanatory, edge, edge_relation, is_main_edge)
    return discourse_tree.function_map[review]


def string_match(string, sub_str):
    # 蛮力法字符串匹配
    for i in range(len(string)-len(sub_str)+1):
        index = i       # index指向下一个待比较的字符
        for j in range(len(sub_str)):
            if string[index] == sub_str[j]:
                index += 1
            else:
                break
            if index-i == len(sub_str):
                return i
    return -1
