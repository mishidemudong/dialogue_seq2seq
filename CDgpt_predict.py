#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:15:00 2020

@author: liang
"""

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from tqdm import tqdm
import sentencepiece as spm
import jieba
import random

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

#config_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf/bert_config.json'
#checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf/gpt_model.ckpt'
#dict_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf/vocab.txt'

config_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf (1)/gpt_config.json'
checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf (1)/gpt_model.ckpt'
dict_path = '/media/liang/Nas/PreTrainModel/gpt/GPT_LCCC-large-tf (1)/vocab.txt'


max_len = 512

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
speakers = [
    tokenizer.token_to_id('[speaker1]'),
    tokenizer.token_to_id('[speaker2]')
]

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='gpt_openai'
)  # 建立模型，加载权重


#class ChatBot1(AutoRegressiveDecoder):
#    """基于随机采样对话机器人
#    """
#    @AutoRegressiveDecoder.wraps(default_rtype='probas')
#    def predict(self, inputs, output_ids, states):
#        token_ids, segment_ids = inputs
#        curr_segment_ids = np.zeros_like(output_ids) + token_ids[0, -1]
#        token_ids = np.concatenate([token_ids, output_ids], 1)
#        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
#        return model.predict([token_ids, segment_ids])[:, -1]
#
#    def response(self, texts, topk=5, test=False):
#        token_ids = [tokenizer._token_start_id, speakers[0]]
#        segment_ids = [tokenizer._token_start_id, speakers[0]]
#        
#        if test:
#            for i, text in enumerate(texts):
#                ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
#                token_ids.extend(ids)
#                segment_ids.extend([speakers[i % 2]] * len(ids))
#                segment_ids[-1] = speakers[(i + 1) % 2]
#        else:
#            i = 0
#            ids = tokenizer.encode(texts, maxlen=510)[0][1:]
#            token_ids += ids
#            segment_ids += [speakers[i % 2]] * len(ids)
#            
#        if len(token_ids) == len(segment_ids):
#        
#            results = self.random_sample([token_ids, segment_ids], 1, topk)
#            return tokenizer.decode(results[0])
#        
#        else:
#            return False

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        curr_segment_ids = np.zeros_like(output_ids) + token_ids[0, -1]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        
#        print("token_ids.shape",token_ids.shape)
#        print("segment_ids.shape",segment_ids.shape)
        
        if len(token_ids[0]) != len(segment_ids[0]):
            
            segment_ids[0] = segment_ids[0][:len(token_ids[0])-1]+[1]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
#        print("token_ids.shape",token_ids.shape)
#        print("segment_ids.shape",segment_ids.shape)
#        print(len(token_ids[0]))
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids = [tokenizer._token_start_id, speakers[0]]
        segment_ids = [tokenizer._token_start_id, speakers[0]]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
            token_ids.extend(ids)
            segment_ids.extend([speakers[i % 2]] * len(ids))
            segment_ids[-1] = speakers[(i + 1) % 2]
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


#chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

class ChatBot2(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=10):
        token_ids = [tokenizer._token_start_id, speakers[0]]
        segment_ids = [tokenizer._token_start_id, speakers[0]]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
            token_ids.extend(ids)
            segment_ids.extend([speakers[i % 2]] * len(ids))
            segment_ids[-1] = speakers[(i + 1) % 2]
            
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)

chatbot = ChatBot2(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def predict():
    
    file_path = '/media/liang/Nas/corpus/Plato2/train_data/test_xianliao_noLccc.txt'
    all_lines = [eval(line) for line in open(file_path, encoding='utf-8')]
    
    pred_data=[]
    for line in tqdm(all_lines[10556:]):
        history = line['history'][0].replace(' ','')[:500] 
    #    if len(history) > 1:
    #        print(history)
#        xiuzheng = []
#        for item in history:
#            xiuzheng.append(item.replace(' ',''))
    #        print(xiuzheng)
        pred_data.append(history)
    
    result_path = './result/xianliao_CDgpt_large_result_noLccc.txt'
    with open(result_path, encoding='utf-8', mode='w') as fr:
        for line in tqdm(pred_data):
            response = chatbot.response([line])

            response = ' '.join(jieba.cut(response))
            fr.writelines(response + '\n')
            
def eval_dev(sampledata):
    
    
    #file_path = '/media/liang/Nas/corpus/Plato2/train_data/dev_xianliao.txt'
    file_path = "/media/liang/Nas/corpus/Plato2/train_data/LCCC/LCCD_dev.json"
    all_lines = [eval(line) for line in open(file_path, encoding='utf-8')]
    
    sampledata = random.sample(all_lines, 1000)
    #eval_result, eval_data = eval_dev(sampledata)
    #print(eval_result)
    
    eval_data=[]
    for line in tqdm(sampledata):
        if 'history' in line.keys():
            history = line['history'].replace(' ','')
            if len(history) > 500:
                history = history[-480:]
    #        print(len(history))
            
            pred = chatbot.response([history])
            if pred:
                gold_tokens = line['response'].split(' ')
                eval_data.append([list(jieba.cut(pred)), gold_tokens])
        elif 'conversation' in line.keys():
            
            conversation = line["conversation"]
                    
            for i in range(1, len(conversation)):
                pred = chatbot.response([item.replace(' ', '') for item in conversation[:i]])
                gold_tokens = conversation[i].split(' ')
                eval_data.append([list(jieba.cut(pred)), gold_tokens])
    
    from bleueval_tool import calc_f1, calc_bleu, calc_distinct
    # calc f1
    f1 = calc_f1(eval_data)
    # calc bleu
    bleu1, bleu2 = calc_bleu(eval_data)
    # calc distinct
    distinct1, distinct2 = calc_distinct(eval_data) 
    
    output_str = "F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f%%\n" % bleu1
    output_str += "BLEU2: %.3f%%\n" % bleu2
    output_str += "DISTINCT1: %.3f%%\n" % distinct1
    output_str += "DISTINCT2: %.3f%%\n" % distinct2
#    
#    output_str2 = "F1: %.2f%%\n" % (f1 * 100)
#    output_str2 += "BLEU1: %.3f%%\n" % bleu1
#    output_str2 += "BLEU2: %.3f%%\n" % bleu2
#    output_str2 += "DISTINCT1: %.3f%%\n" % distinct1
#    output_str2 += "DISTINCT2: %.3f%%\n" % distinct2
    
    return output_str, eval_data


#predict()
file_path = '/media/liang/Nas/corpus/Plato2/train_data/test_xianliao_noLccc.txt'
all_lines = [eval(line) for line in open(file_path, encoding='utf-8')]

pred_data=[]
for line in tqdm(all_lines): #10555
    history = line['history'][0].replace(' ','')
    if len(history) > 500:
        history = history[-480:]
#    if len(history) > 1:
#        print(history)
#        xiuzheng = []
#        for item in history:
#            xiuzheng.append(item.replace(' ',''))
#        print(xiuzheng)
    pred_data.append(history)

result_path = './result/xianliao_CDgpt_large_result_noLccc_beamsearch.txt'
with open(result_path, encoding='utf-8', mode='w') as fr:
    for text in tqdm(pred_data):
        response = chatbot.response([text])

        response = ' '.join(jieba.cut(response))
        fr.writelines(response + '\n')

    
    
    