#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:19:57 2020

@author: liang
"""


import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

max_goal_len = 64
max_kg_len = 256
max_con_len = 192

max_res_len = 128

batch_size = 2
steps_per_epoch = 1000
epochs = 10000

# nezha配置
config_path = '/media/liang/Nas/PreTrainModel/gpt/NEZHA-Base/bert_config.json'
checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/NEZHA-Base/model.ckpt-900000'
dict_path = '/media/liang/Nas/PreTrainModel/gpt/NEZHA-Base/vocab_kg.txt'

#bert_path = '/media/liang/Nas/PreTrainModel'
#config_path = bert_path + '/albert/albert_tiny_zh_google/albert_config_tiny_g.json' #albert_xlarge_google_zh_183k
#checkpoint_path = bert_path + '/albert/albert_tiny_zh_google/albert_model.ckpt'
#dict_path = bert_path + '/albert/albert_tiny_zh_google/vocab_kg.txt'


# bert_path = '/media/liang/Nas/PreTrainModel'
# config_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json' #albert_xlarge_google_zh_183k
# checkpoint_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab_kg.txt'


# config_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/config.json'
# checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/model.ckpt'
# dict_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/vocab_kg.txt'

#data_type = {"[GOAL]": , "[KG]":}

def load_data(filename, number=None, is_test=False):
    D = []
    count = 0
    with open(filename, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
#            print(data)

            goal = data["goal"] if "goal" in data.keys() else ""
            goal = ' '.join([' '.join(spo) for spo in goal])
            
#            if "knowledge" in data.keys():
#                #random sample 20 spos
#                knowledge = list(set([' '.join(spo) for spo in data["knowledge"]]))
#                if len(knowledge) > 8:
#                    knowledge = random.sample(knowledge, 8)
#                
#                knowledge = ' '.join(knowledge) 
#            else:
#                knowledge = ""
            knowledge = list(set([' '.join(spo) for spo in data["knowledge"]]))
#            print("****",knowledge)
            conversation = data["conversation"] if "conversation" in data.keys() else ""
#            print("====",conversation)
                
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

# 加载数据集
kg_train_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/train_kg_notencent.txt')
kg_valid_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/dev_kg_notencent.txt')
#xianliao_train = load_data('/media/liang/Nas/corpus/Plato2/input/LCCC/LCCC/LCCD_train.json', number=100000)
#xianliao_dev = load_data('/media/liang/Nas/corpus/Plato2/input/LCCC/LCCC/LCCD_dev.json', number=20000)
#tengxun_train = load_data('/media/liang/Nas/corpus/Plato2/train_data/tencent/train.txt', number=100000)
#tengxun_dev = load_data('/media/liang/Nas/corpus/Plato2/train_data/tencent/dev.txt', number=20000)
#train_data = xianliao_train + kg_train_data + tengxun_train#+ train_data2 + train_data3
#valid_data = xianliao_dev + kg_valid_data + tengxun_dev #+ valid_data2 + valid_data3

train_data = kg_train_data #+ tengxun_train#+ train_data2 + train_data3
valid_data = kg_valid_data #+ tengxun_dev #+ valid_data2 + valid_data3

# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[GOAL]', '[KG]'],
#    startswith=['[PAD]', '[GOAL]', '[KG]', '[UNK]', '[CLS]', '[SEP]'],
)

# 补充词表
compound_tokens = []
for l in open('user_tokens.csv', encoding='utf-8'):
    token, count = l.strip().split('\t')
    if int(count) >= 10 and token not in token_dict:
        token_dict[token] = len(token_dict)
        compound_tokens.append([0])

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)

print_step = 0
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        global print_step
        for is_end, sample in self.sample(random):
            
            goal_token_ids, goal_seg_ids = tokenizer.encode(sample['goal'], maxlen=max_goal_len-1)
            kg_token_ids, kg_seg_ids = tokenizer.encode(sample['knowledge'], maxlen=max_kg_len-1)

            token_ids, segment_ids = [tokenizer.token_to_id('[CLS]')] + [tokenizer.token_to_id('[GOAL]')] +goal_token_ids[1:] + [tokenizer.token_to_id('[KG]')] +kg_token_ids[1:], [0] + goal_seg_ids + [0] + kg_seg_ids[1:]
            
            for i, text in enumerate(sample['context']):
                ids = tokenizer.encode(text)[0][1:]
                if len(token_ids) + len(ids) <= max_con_len:
                    token_ids.extend(ids)
                    segment_ids.extend([i % 2] * len(ids))
                else:
                    break
            
            if len(token_ids) > 5:
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                if print_step % 500 ==0:
                    print()
                    print("token_ids:", tokenizer.ids_to_tokens(token_ids))
                    print("token_:", tokenizer.decode(token_ids))
                print_step += 1
            
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 要扩充的词表
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])
model = Model(model.inputs, output)
#model.load_weights('./latest_model_nezha.weights')

model.summary()


AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=2
)
model.compile(optimizer=optimizer)


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


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def on_epoch_end(self, epoch, logs=None):
        while True:
            try:
                model.save_weights('./latest_model_nezha.weights')
                break
            except:
                print(u'保存失败，正在重试...')

class EvaluatorA(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 5:
            metrics = self.evaluate(valid_data)  # 评测模型
            if metrics['bleu'] > self.best_bleu:
                self.best_bleu = metrics['bleu']
                model.save_weights('./latest_model_nezha.weights')  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for sample in tqdm(data):
            total += 1
            real = ' '.join(sample['response']).lower()
            response = ' '.join(chatbot.response(sample, topk)).lower()
            
            if total%100==0:
                print()
                print("real:", real)
                print("pred:", response)
            if response.strip():
                scores = self.rouge.get_scores(hyps=response, refs=real)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[real.split(' ')],
                    hypothesis=response.split(' '),
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

    evaluator = EvaluatorA()
    
    train_data = train_data#[:1000]
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./latest_model_nezha.weights')
