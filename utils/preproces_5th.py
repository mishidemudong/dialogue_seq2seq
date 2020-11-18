#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:55:20 2020

@author: liang
"""
import json
from tqdm import tqdm


def _compute_overlap_num(tgt, kg):
    if isinstance(tgt, str):
        tgt = list(set(tgt.split()))
    if isinstance(kg, str):
        kg = list(set(kg.split()))
    overlap_num = 0
    for t in tgt:
        for k in kg:
            if t == k:
                overlap_num +=1
    return overlap_num


def get_words(kg):
    words = []
    for item in kg:
        for w in item.split():
            words.append(w)
    return words


def knowledge_select(tgt, knowledge):
    valid_knowledge = []
    for kg in knowledge:
        kg = get_words(kg)
        # print(kg)
        _kg = " ".join(kg)
        overlap_num = _compute_overlap_num(tgt, _kg)
        if overlap_num > 0:
            valid_knowledge.append(kg)
    return valid_knowledge

def make_examples(path):
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line.strip("\n"))
            yield data

    fr.close()


def goal_process(goal):
    input_from_goal = []
    for ix, g in enumerate(goal):
        input_from_goal.append("[Goal-%d]" %ix)
        for token in g:
            input_from_goal.append(token)
    return input_from_goal


def knowledge_process(knowledge):
    input_from_knowledge = []
    for k in knowledge:
        input_from_knowledge.append("[K-G]")
        for token in k:
            input_from_knowledge.append(token)
    # input_from_knowledge.append("[SEP]")
    return input_from_knowledge


def conversation_process(conversation):
    input_from_conversation = []
    target_from_conversation = []
    turns = 1
    dialogue = []
    dialogue.append("[Session-0]")
    for index, conver in enumerate(conversation):
        if index % 2 == 1:
            dialogue.append("[Session-%d]"%turns)
            dialogue.append(conver)
            turns += 1
        else:
            dialogue.append(conver)

    ix = 0
    while ix < len(dialogue):
        if ix % 2 == 1:
            target_from_conversation.append(dialogue[ix])
            input_from_conversation.append((dialogue[max(0, ix-5): ix], dialogue[max(0, ix-3): ix])) # 保留前两轮对话

        ix += 1
    return input_from_conversation, target_from_conversation


def single_example_process(example):
    goal = example.get("goal", [])
    knowledge = example.get("knowledge", [])
    conversation = example.get("conversation", [])
    input_from_goal = goal_process(goal)

    input_from_conversation, target_from_conversation = conversation_process(conversation)
    src, tgt = [], []
    for conver, target in zip(input_from_conversation, target_from_conversation):
        for c in conver:
            valid_knowledge = knowledge_select(target, knowledge)
            input_from_knowledge = knowledge_process(valid_knowledge)

#            inputs = " ".join(c) + " " + " ".join(input_from_knowledge) + " " + " ".join(input_from_goal)
            # inputs = ltp_pos(inputs)
#            src.append(inputs)
            src.append({
                        'conversation':c,
                        'knowledge':input_from_knowledge,
                        'goal':input_from_goal
                    }
                    )
            
            
            tgt.append(target)
    # for s, t in zip(src, tgt):
    #     print("A: ", s)
    #     print("B: ", t)
    return src, tgt

def goal_select():
    pass


def drop_duplicate(src, tgt):
    src_dict = {}
    new_src = []
    new_tgt=  []
    for s, t in zip(src, tgt):
        if src_dict.get(s) is None:
            new_src.append(s)
            new_tgt.append(t)
            src_dict[s] = 1
    return new_src, new_tgt

def load_data(filepath):
    D=[]
    examples = make_examples(filepath)
    for example in tqdm(examples):
        src, tgt = single_example_process(example)     
        for x,y in zip(src, tgt):
            D.append((x, y))
            
    return D

if __name__ == '__main__':
    raw_data_path = "/media/liang/Nas/corpus/Plato2/train_data/train_kg_2.txt"
    
    
    examples = make_examples(raw_data_path)
    
    for example in tqdm(examples):
        src, tgt = single_example_process(example)
    #    src, tgt = drop_duplicate(src, tgt)
        
        break