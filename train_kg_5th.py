#! -*- coding: utf-8 -*-
# NEZHA模型做闲聊任务
# 训练脚本
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.4

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Input
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
import jieba
import random
from utils.preproces_5th import load_data

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

max_kg_len = 388
max_con_len = 388
max_res_len = 48

batch_size = 4
steps_per_epoch = 1000
epochs = 10000

# nezha配置
#config_path = '/media/liang/Nas/PreTrainModel/gpt/chinese_wonezha_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/chinese_wonezha_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '/media/liang/Nas/PreTrainModel/gpt/chinese_wonezha_L-12_H-768_A-12/vocab_kg_5.txt'

#config_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/config.json'
#checkpoint_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/model.ckpt'
#dict_path = '/media/liang/Nas/PreTrainModel/gpt/nezha_gpt_dialog/vocab_kg.txt'

###roberta
bert_path = '/media/liang/Nas/PreTrainModel'
config_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json' 
checkpoint_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = bert_path + '/roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab_kg5.txt'


  




# 加载数据集
train_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/train_kg_notencent.txt')
valid_data = load_data('/media/liang/Nas/corpus/Plato2/train_data/dev_kg_notencent.txt') 
valid_data = random.sample(valid_data, 800)


special_tokens = ['[Goal-0]','[Goal-1]', '[Goal-2]', '[K-G]', '[Session-0]', '[Session-1]', '[Session-2]', '[Session-3]', '[Session-4]', '[Session-5]', '[Session-6]', '[Session-7]', '[Session-8]', '[Session-9]']
# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + special_tokens
    )

# 建立分词器
#tokenizer = Tokenizer(
#    token_dict,
#    do_lower_case=True,
#    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
#)

######
class LMTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super(LMTokenizer, self).__init__(*args, **kwargs)
        for token in special_tokens:
            _token_id = self._token_dict[token]
            setattr(self, '_token_{}_id'.format(token.lstrip('[').rstrip(']').lower()), _token_id)
            
tokenizer = LMTokenizer(token_dict, do_lower_case=True)
        
print_step = 0
class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        global print_step
        for is_end, (data_info, response) in self.sample(train_data):
    
            all_token_ids = [tokenizer.token_to_id('[CLS]')]
            all_seg_ids = [0]
            
            kg_ids = []
            kg_seg = []
            for kg in data_info['knowledge']:
                if kg in special_tokens:
                    token_ids = [tokenizer.token_to_id(kg)]
                else:
                    token_ids = tokenizer.encode(kg, maxlen=max_kg_len)[0][1:-1]
                
                if len(kg_ids) + len(token_ids) < max_kg_len:
                    kg_ids.extend(token_ids)
                    kg_seg.extend([0]*len(token_ids))
                
            goal_ids = []
            goal_seg = []
            for goal in data_info['goal']:
                if goal in special_tokens:
                    token_ids = [tokenizer.token_to_id(goal)]
                else:
                    token_ids = tokenizer.encode(goal)[0][1:-1]
                goal_ids.extend(token_ids)
                goal_seg.extend([0]*len(token_ids))
            
            conva_ids = []
            conva_seg = []
            current_turn = 1
            for index, conv in enumerate(data_info['conversation']):
                if conv in special_tokens:
                    token_ids = [tokenizer.token_to_id(conv)]
                    current_turn = int(conv[9]) + 1
                else:
                    token_ids = tokenizer.encode(conv)[0][1:-1]
                
                if len(conva_ids) + len(token_ids) < max_con_len:
                    conva_ids.extend(token_ids)
                    conva_seg.extend([current_turn]*len(token_ids))
            
            if response.startswith('[Session'):
                continue
            
            res_token_ids = tokenizer.encode(response, maxlen=max_res_len)[0][1:]
            res_seg_ids = [current_turn]*len(res_token_ids)
            
            tmp_ids = goal_ids + kg_ids + conva_ids + res_token_ids
            tmp_seg = goal_seg + kg_seg + conva_seg + res_seg_ids
            if len(tmp_ids) < 512:
                all_token_ids.extend(tmp_ids)
                all_seg_ids.extend(tmp_seg)
            else:
                all_token_ids.extend(tmp_ids[-511:])
                all_seg_ids.extend(tmp_seg[-511:])
                
            batch_token_ids.append(all_token_ids)
            batch_segment_ids.append(all_seg_ids)
                
            if print_step % 500 ==0:
                print()
                print(print_step)
                print("token_ids:", tokenizer.ids_to_tokens(all_token_ids))
                print("token_decode:", tokenizer.decode(all_token_ids))
                print("seg:",all_seg_ids)
            print_step += 1

            if len(batch_token_ids) == batch_size or is_end:
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



#model = build_transformer_model(
#    config_path,
#    checkpoint_path,
#    model='nezha',
#    application='lm',
#    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
##    compound_tokens=compound_tokens,  # 要扩充的词表
#)

#output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])
#model = Model(model.inputs, output)
#model.summary()

roberta_model = build_transformer_model(
            config_path,
            checkpoint_path,
            application='lm',
            keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        )

output = CrossEntropy(1)([roberta_model.inputs[0], roberta_model.outputs[0]])
model = Model(roberta_model.inputs, output)

model.summary()


model.compile(optimizer=Adam(2e-4))
#AdamW = extend_with_weight_decay(Adam, 'AdamW')
#AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
#optimizer = AdamWG(
#    learning_rate=2e-5,
#    weight_decay_rate=0.01,
#    exclude_from_weight_decay=['Norm', 'bias'],
#    grad_accum_steps=4
#)
#model.compile(optimizer=optimizer)


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

    def response(self, data_info, topk=5):
        all_token_ids = [tokenizer.token_to_id('[CLS]')]
        all_seg_ids = [0]
        kg_ids = []
        kg_seg = []
        for kg in data_info['knowledge']:
            if kg in special_tokens:
                token_ids = [tokenizer.token_to_id(kg)]
            else:
                token_ids = tokenizer.encode(kg, maxlen=max_kg_len)[0][1:-1]
            
            if len(kg_ids) + len(token_ids) < max_kg_len:
                kg_ids.extend(token_ids)
                kg_seg.extend([0]*len(token_ids))
            
        goal_ids = []
        goal_seg = []
        for goal in data_info['goal']:
            if goal in special_tokens:
                token_ids = [tokenizer.token_to_id(goal)]
            else:
                token_ids = tokenizer.encode(goal)[0][1:-1]
            goal_ids.extend(token_ids)
            goal_seg.extend([0]*len(token_ids))
        
        conva_ids = []
        conva_seg = []
        current_turn = 1
        for index, conv in enumerate(data_info['conversation']):
            if conv in special_tokens:
                token_ids = [tokenizer.token_to_id(conv)]
                current_turn = int(conv[9]) + 1
            else:
                token_ids = tokenizer.encode(conv)[0][1:-1]
            
            if len(conva_ids) + len(token_ids) < max_con_len:
                conva_ids.extend(token_ids)
                conva_seg.extend([current_turn]*len(token_ids))
        
        
        all_token_ids.extend(goal_ids + kg_ids + conva_ids)
        all_seg_ids.extend(goal_seg + kg_seg + conva_seg)
        
        results = self.random_sample([all_token_ids, all_seg_ids], 1, topk)
        return tokenizer.decode(results[0])

chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=max_res_len)


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
                model.save_weights('./wo_model_nezha.weights')  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for sample in tqdm(data):
#            print(sample[0])
            total += 1
            real = ' '.join(sample[1]).lower()
            response = ' '.join(chatbot.response(sample[0], topk)).lower()
            
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
    
#    train_data = train_data[:10]
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./latest_model_nezha.weights')
