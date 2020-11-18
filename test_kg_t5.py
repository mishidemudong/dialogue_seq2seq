#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:55:49 2020

@author: liang
"""


from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, :-1]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def load_data(filename, is_test=False):
    D = []
    count = 0
    with open(filename, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
#            print(data)

            goal = data["goal"] if "goal" in data.keys() else ""
            
            goal = ' '.join([' '.join(spo) for spo in goal])

            knowledge = ' '.join([' '.join(spo) for spo in data["knowledge"]]) 
            
            len_goal = len(goal)
            len_kg = len(knowledge)
                
            if not is_test:
                
                
                conversation = data["conversation"] if "conversation" in data.keys() else ""
                
                for i in range(0, len(conversation), 2):
                    sample = {"goal": goal ,
                              "knowledge": knowledge,
                              "context": '\t'.join(conversation[:i]) if i > 0 else "",
                              "response": conversation[i]}
                    
#                    print(sample)
    
                    D.append(sample)        
                    
            else:
                history = data["history"]
                response = data["response"] if "response" in data else ""

                sample = {"goal": goal,
                          "knowledge": knowledge ,
                          "context": '\t'.join(history),
                          "response": response}
                D.append(sample)
            count += 1
            
#            if count == 10000:
#                break
                
    return D

# 基本参数
max_goal_len = 64
max_kg_len = 256
max_con_len = 192
max_res_len = 128


# 模型路径
config_path = '/media/liang/Nas/PreTrainModel/T5/mt5_small/mt5_small_config.json'
checkpoint_path = '/media/liang/Nas/PreTrainModel/T5/mt5_small/model.ckpt-1000000'
spm_path = '/media/liang/Nas/PreTrainModel/T5/sentencepiece_cn.model'
keep_tokens_path = '/media/liang/Nas/PreTrainModel/T5/sentencepiece_cn_keep_tokens.json'


# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))

# 建立并加载模型
t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model

model.summary()

output = CrossEntropy(1)([model.inputs[1], model.outputs[0]])

model = Model(model.inputs, output)

model.load_weights('./best_model_t5.weights')

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder.predict([c_encoded, output_ids])[:, -1]

    def generate(self, sample, topk=1):
        goal_token_ids, _ = tokenizer.encode(sample['goal'], maxlen=max_goal_len)
        kg_token_ids, _ = tokenizer.encode(sample['knowledge'], maxlen=max_kg_len)
        con_token_ids, _ = tokenizer.encode(sample['context'], maxlen=max_con_len)
        token_ids = goal_token_ids + kg_token_ids[1:] + con_token_ids[1:]
        
        
        c_encoded = encoder.predict(np.array([token_ids]))[0]
#        print(c_encoded)
        output_ids = self.beam_search([c_encoded], topk)  # 基于beam search
#        print(type(output_ids))
        return tokenizer.decode(output_ids.tolist())


# 注：T5有一个很让人不解的设置，它的<bos>标记id是0，即<bos>和<pad>其实都是0
autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=64)



if __name__ == '__main__':
#    test()
    
    testdata = load_data("/media/liang/Nas/corpus/Plato2/train_data/test_kg.txt", True)
#    print(autotitle.generate(testdata[0]))
    
    count = 0
    result_path = './result/kg_pred_1116.txt'
    with open(result_path, encoding='utf-8', mode='w') as fr:

        for data in tqdm(testdata):
#            print()
#            print("histtory:", data['context'])
#            
#            print("response:", autotitle.generate(data))
            
            
            
            fr.writelines(autotitle.generate(data) + '\n')
            count += 1
            if count > 100:
                fr.close()
                break
        
        


