#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:10:33 2020

@author: liang
"""
import random

def sampleKG(knowledge):
    knowledge = [' '.join(spo) for spo in knowledge]
    if len(knowledge) > 30:
        knowledge = random.sample(knowledge, 30)
        knowledge = ' '.join(knowledge) 




















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
        print("kg",kg)
        _kg = " ".join(kg)
        overlap_num = _compute_overlap_num(tgt, _kg)
        if overlap_num > 0:
            valid_knowledge.append(kg)
    return valid_knowledge