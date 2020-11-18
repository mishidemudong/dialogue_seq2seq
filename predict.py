#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:23:51 2020

@author: liang
"""


import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from tqdm import tqdm
import random
import jieba

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# nezha配置
config_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/config.json'
checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/model.ckpt'
dict_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 建立并加载模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
)
#model.load_weights('./latest_model_nezha.weights')
model.summary()


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)
#print(chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样']))
"""
回复是随机的，例如：那你还爱我吗 | 不知道 | 爱情是不是不能因为一点小事就否定了 | 我会一直爱你，你一个人会很辛苦 | 等等。
"""
file_path = '/media/liang/Nas/corpus/Plato2/train_data/LCCC/test.txt'
all_lines = [eval(line) for line in open(file_path, encoding='utf-8')]

pred_data=[]
for line in tqdm(all_lines):
    history = [item.replace(' ','') for item in line['history']]
#    if len(history) > 1:
#        print(history)
#    xiuzheng = []
#    for item in history:
#        xiuzheng.append(item.replace(' ',''))
#        print(xiuzheng)
    pred_data.append(history)

result_path = './result/xianliao_lccc_result.txt'
with open(result_path, encoding='utf-8', mode='w') as fr:
    for line in tqdm(pred_data):
        response = chatbot.response(line)
        fr.writelines(response + '\n')
        
        
#file_path = '/media/liang/Nas/corpus/Plato2/train_data/dev_xianliao.txt'
#file_path = "/media/liang/Nas/corpus/Plato2/train_data/LCCC/LCCD_dev.json"
#
#all_lines = [eval(line) for line in open(file_path, encoding='utf-8')]
#
#sampledata = random.sample(all_lines, 500)
##eval_result, eval_data = eval_dev(sampledata)
##print(eval_result)
#
#eval_data=[]
#for line in tqdm(sampledata):
#    if 'history' in line.keys():
#        history = line['history'].replace(' ','') 
#        
#        pred = chatbot.response([history])
#        gold_tokens = line['response'].split(' ')
#        eval_data.append([list(jieba.cut(pred)), gold_tokens])
#    elif 'conversation' in line.keys():
#        
#        conversation = line["conversation"]
#                
#        for i in range(1, len(conversation)):
#
#            pred = chatbot.response([item.replace(' ', '') for item in conversation[:i]])
#            gold_tokens = conversation[i].split(' ')
#            eval_data.append([list(jieba.cut(pred)), gold_tokens])
#
#from bleueval_tool import calc_f1, calc_bleu, calc_distinct
## calc f1
#f1 = calc_f1(eval_data)
## calc bleu
#bleu1, bleu2 = calc_bleu(eval_data)
## calc distinct
#distinct1, distinct2 = calc_distinct(eval_data)
#output_str2 = "F1: %.2f%%\n" % (f1 * 100)
#output_str2 += "BLEU1: %.3f%%\n" % bleu1
#output_str2 += "BLEU2: %.3f%%\n" % bleu2
#output_str2 += "DISTINCT1: %.3f%%\n" % distinct1
#output_str2 += "DISTINCT2: %.3f%%\n" % distinct2
