#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:01:41 2020

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
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

from utils.kgdata_clean import knowledge_select


import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 基本参数
max_goal_len = 24
max_kg_len = 192
max_con_len = 160

max_res_len = 32
batch_size = 16
epochs = 40


# 模型路径
config_path = '/media/liang/Nas/PreTrainModel/T5/mt5_small/mt5_small_config.json'
checkpoint_path = '/media/liang/Nas/PreTrainModel/T5/mt5_small/model.ckpt-1000000'
spm_path = '/media/liang/Nas/PreTrainModel/T5/sentencepiece_cn.model'
keep_tokens_path = '/media/liang/Nas/PreTrainModel/T5/sentencepiece_cn_keep_tokens.json'

goallen_dict = {}
kglen_dict = {}
conlen_dict = {}

def load_data(filename, number=None, is_test=False):
    D = []
    count = 0
    with open(filename, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
#            print(data)

            goal = data["goal"] if "goal" in data.keys() else ""
            goal = ' '.join([' '.join(spo) for spo in goal])
            
            if "knowledge" in data.keys():
                #random sample 20 spos
                knowledge = list(set([' '.join(spo) for spo in data["knowledge"]]))
                new_kg = []
                for item in knowledge:
                    if len(item) < 64:
                        new_kg.append(item)
                    else:
                        new_kg.append(item[:64])
                    
                if len(new_kg) > 8:
                    knowledge = random.sample(new_kg, 8)
                else:
                    knowledge = new_kg
                
                knowledge = ' '.join(knowledge) 
            else:
                knowledge = ""
                
#            print("****",knowledge)
            conversation = data["conversation"] if "conversation" in data.keys() else ""
#            print("====",conversation)
            
            #according to dialogue
#            knowledge = knowledge_select(conversation, knowledge)
            
            len_goal = len(goal)
            len_kg = len(knowledge)
            
            if  len_goal not in goallen_dict:
                goallen_dict[len_goal]=1
            else:
                goallen_dict[len_goal]=goallen_dict[len_goal]+1
            
            if len_kg not in kglen_dict:
                kglen_dict[len_kg]=1
            else:
                 kglen_dict[len_kg] = kglen_dict[len_kg]+1
                
            if not is_test:
                for i in range(0, len(conversation), 2):
                    sample = {"goal": goal ,
                              "knowledge": knowledge,
                              "context": conversation[:i] if i > 0 else "",
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
            
            if count == number:
                break
                
    return D


def load_data_ori(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            reponse, question = l.strip().split('\t')
            D.append((reponse, question))
    return D


# 加载数据集
kg_train_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/train_kg_notencent.txt')
kg_valid_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/dev_kg_notencent.txt') 

#xianliao_train = load_data('/media/liang/Nas/corpus/Plato2/input/LCCC/LCCC/LCCD_train.json', number=100000)
#xianliao_dev = load_data('/media/liang/Nas/corpus/Plato2/input/LCCC/LCCC/LCCD_dev.json', number=20000)

#tengxun_train = load_data('/media/liang/Nas/corpus/Plato2/train_data/tencent/train.txt', number=100000)
#tengxun_dev = load_data('/media/liang/Nas/corpus/Plato2/train_data/tencent/dev.txt', number=20000)


#print(sorted(goallen_dict.items(), key=lambda x: x[0], reverse=False))
#print(sorted(kglen_dict.items(), key=lambda x: x[0], reverse=False))

#print(sum(kglen_dict.values())/len(kglen_dict.values()))
#3206.7027334851937

#print(sum(kglen_dict.keys())/len(kglen_dict.keys()))
#1153.73861047836


#train_data = xianliao_train + kg_train_data + tengxun_train#+ train_data2 + train_data3
#valid_data = xianliao_dev + kg_valid_data + tengxun_dev #+ valid_data2 + valid_data3
#valid_data = random.sample(valid_data, 8000)
#train_data = train_data#[:100]
train_data = kg_train_data #+ tengxun_train#+ train_data2 + train_data3
valid_data = kg_valid_data #+ tengxun_dev #+ valid_data2 + valid_data3
valid_data = random.sample(valid_data, 3000)


# 加载分词器
tokenizer = SpTokenizer(spm_path) #, token_start=None, token_end='</s>'
keep_tokens = json.load(open(keep_tokens_path))

#import sentencepiece as spm
#tokenizer = spm.SentencePieceProcessor()
#tokenizer.load("/media/liang/Project2/CCF-Baidu/my_Knover/luge-dialogue/config/spm.model")

print_step = 0
class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random=False):
        batch_con_token_ids, batch_res_token_ids = [], []
        global print_step
        for is_end, sample in self.sample(random):
            
            goal_token_ids, _ = tokenizer.encode(sample['goal'], maxlen=max_goal_len)
            kg_token_ids, _ = tokenizer.encode(sample['knowledge'], maxlen=max_kg_len)
            
            context_list= sample['context']
#            con_token_ids = [tokenizer._token_start_id]
#            for context in context_list:
#                ids = tokenizer.encode(context)[0][1:]
#                if len(con_token_ids) + len(ids) <= max_con_len:
#                    con_token_ids.extend(ids)
#                else:
#                    break
                
            con_token_ids, _ = tokenizer.encode('\t'.join(context_list), maxlen=max_con_len)
            
            res_token_ids, _ = tokenizer.encode(sample['response'], maxlen=max_res_len)
            
#            print("goal",goal_token_ids)
#            print()
#            print("kg",kg_token_ids)
#            print()
#            print("goal",con_token_ids)
            token_ids = goal_token_ids + kg_token_ids[1:] + con_token_ids[1:]
            
#            print()
#            print("token",token_ids)
            
            if len(token_ids) > 4:
                batch_con_token_ids.append(token_ids)
                batch_res_token_ids.append(res_token_ids)
                if print_step % 1000 ==0:
                    print()
                    print("token_ids:", tokenizer.ids_to_tokens(token_ids))
                    print("token_:", tokenizer.decode(token_ids))
                print_step += 1
            
            if len(batch_con_token_ids) == self.batch_size or is_end:
                batch_con_token_ids = sequence_padding(batch_con_token_ids)
                batch_res_token_ids = sequence_padding(batch_res_token_ids)
                yield [batch_con_token_ids, batch_res_token_ids], None
                batch_con_token_ids, batch_res_token_ids = [], []


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
#model.load_weights('./best_model_t5.weights')
model.compile(optimizer=Adam(2e-4))

#AdamW = extend_with_weight_decay(Adam, 'AdamW')
#AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
#optimizer = AdamWG(
#    learning_rate=2e-4,
#    weight_decay_rate=0.01,
#    exclude_from_weight_decay=['Norm', 'bias'],
#    grad_accum_steps=16
#)
#model.compile(optimizer=optimizer)


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
autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=48)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 2:
            metrics = self.evaluate(valid_data)  # 评测模型
            if metrics['bleu'] > self.best_bleu:
                self.best_bleu = metrics['bleu']
                model.save_weights('./best_model_t5_short_last.weights')  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)

    def evaluate(self, data, topk=5):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for sample in tqdm(data):
            total += 1
            response = ' '.join(sample['response']).lower()
            pred_title = ' '.join(autotitle.generate(sample, topk)).lower()
            
            if total%10==0:
                print()
                print("context:", sample['context'])
                print("real:", response)
                print("pred:", pred_title)
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=response)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[response.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }
        



if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
    
    rouge = Rouge()
    data = valid_data
    topk = 5
    smooth = SmoothingFunction().method1
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for sample in tqdm(data):
        total += 1
        response = ' '.join(sample['response']).lower()
        pred_title = ' '.join(autotitle.generate(sample, topk)).lower()
        
        if total%10==0:
            print()
            print("context:", sample['context'])
            print("real:", response)
            print("pred:", pred_title)
        if pred_title.strip():
            scores = rouge.get_scores(hyps=pred_title, refs=response)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[response.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
#        return {
#            'rouge-1': rouge_1,
#            'rouge-2': rouge_2,
#            'rouge-l': rouge_l,
#            'bleu': bleu,
#        }
    
    print( {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    })
