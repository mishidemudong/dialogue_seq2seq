#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:12:08 2020

@author: liang
"""


import os, sys

from utils.base_input import *


def trans(data_type):
#    data_dict = {
#        0: join(DATA_PATH, 'train/train.txt'),
#        1: join(DATA_PATH, 'dev/dev.txt'),
#        2: join(DATA_PATH, 'test_1/test_1.txt'),
#        3: join(DATA_PATH, 'test_2/test_2.txt'),
#    }
    
    data_dict = {
        0: '/media/liang/Nas/corpus/Plato2/train_data/train_kg_notencent.txt',
        1: '/media/liang/Nas/corpus/Plato2/train_data/dev_kg_notencent.txt',
        2: '/media/liang/Nas/corpus/Plato2/train_data/test_kg_notencent.txt',
        3: 'test_2/test_2.txt',
    }
    
    output_dir =  'trans'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = join(output_dir, 'trans_{}.txt'.format(data_type))

    data_input = BaseInput()

    all_data = []

    data_iter = data_input.get_sample(data_dict[data_type], need_shuffle=False, cycle=False)
    sn = 0
    for sample in data_iter:
        context, goals, turns, unused_goals, replace_dicts = data_input.reader.trans_sample(
            sample, return_rest_goals=True, need_replace_dict=True)
        sample.update(
            {
                'context': context,
                'goals': goals,
                'turns': turns,
                'unused_goals': unused_goals,
                'replace_dicts': replace_dicts,
            }
        )
        all_data.append(sample)
        sn += 1
        if sn % 58 == 0:
            print('\rnum {}'.format(sn), end='  ')
        # if sn > 30:
        #     break
    print('\nOver: ', sn)
    with open(output_path, encoding='utf-8', mode='w') as fw:
        for data in all_data:
            fw.writelines(json.dumps(
                data,
                ensure_ascii=False,
                # indent=4, separators=(',',':')
            ) + '\n')



if __name__ == '__main__':
    trans(0)
#    trans(1)
#    trans(2)
#    trans(3)