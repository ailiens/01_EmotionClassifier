# BERT 사용을 위함
import torch
from transformers import BertTokenizer
# for padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from data_Preprocess2 import train_sentences, validation_sentences, test_sentences, train_labels, validation_labels, test_labels
from classifier import data_Preprocess1
# BERT tokenizer를 이용한 전처리

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

MAX_LEN = 128
def data_to_tensor(sentences, labels):
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating='post', padding='post')

  attention_masks = []

  for seq in input_ids:
      seq_mask = [float(i > 0) for i in seq]
      attention_masks.append(seq_mask)

  tensor_inputs = torch.tensor(input_ids)
  tensor_labels = torch.tensor(labels)
  tensor_masks = torch.tensor(attention_masks)

  return tensor_inputs, tensor_labels, tensor_masks

train_inputs, train_labels, train_masks = data_to_tensor(data_Preprocess1.train_sentences, data_Preprocess1.train_labels)
valid_inputs, valid_labels, valid_masks = data_to_tensor(data_Preprocess1.valid_sentences, data_Preprocess1.valid_labels)
test_inputs, test_labels, test_masks = data_to_tensor(data_Preprocess1.test_sentences, data_Preprocess1.test_labels)

# print(type(train_labels))
# print(type(valid_labels))
# print(type(test_labels))
#
# print(data_Preprocess1.test_sentences[0])
# print(train_inputs[1])
# print(train_masks[0])
# print(train_labels)

# 배치크기 = 32 파이토치 데이터로더(배치 단위로 데이터를 꺼내올 수 있돌록 하는 모듈로 변환)
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

print(test_inputs.shape)
# import pickle
# with open('./data/test_dataloader.pkl','wb') as f:
#   pickle.dump(test_dataloader, f)

# print('train_data:', len(train_labels))
# print('valid_data:', len(valid_labels))
# print('test_data:', len(test_labels))
#
# print(tokenizer.decode([2]))

# print(train_inputs[0])
# print(train_masks[0])
#
# print('훈련 데이터의 크기:', len(train_labels))
# print('검증 데이터의 크기:', len(data_Preprocess2.validation_labels))
# print('테스트 데이터의 크기:', len(test_labels))






















############# 되는거 ######
# tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
#
# MAX_LEN = 128
# def data_to_tensor(sentences, labels):
#   tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
#   input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
#   input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating='post', padding='post')
#
#   attention_masks = []
#
#   for seq in input_ids:
#       seq_mask = [float(i > 0) for i in seq]
#       attention_masks.append(seq_mask)
#
#   tensor_inputs = torch.tensor(input_ids)
#   tensor_labels = torch.tensor(labels)
#   tensor_masks = torch.tensor(attention_masks)
#
#   return tensor_inputs, tensor_labels, tensor_masks
#
# train_inputs, train_labels, train_masks = data_to_tensor(data_Preprocess1.train_sentences, data_Preprocess1.train_labels)
# valid_inputs, valid_labels, valid_masks = data_to_tensor(data_Preprocess1.validation_sentences, data_Preprocess1.validation_labels)
# test_inputs, test_labels, test_masks = data_to_tensor(data_Preprocess1.test_sentences, data_Preprocess1.test_labels)
#
# # print(type(train_labels))
# # print(type(valid_labels))
# # print(type(test_labels))
#
# # print(test_sentences[:5])
# # print(train_inputs[1])
# # print(train_masks[0])
# # print(train_labels)
#
# # 배치크기 = 32 파이토치 데이터로더(배치 단위로 데이터를 꺼내올 수 있돌록 하는 모듈로 변환)
# batch_size = 32
#
# train_data = TensorDataset(train_inputs, train_masks, train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#
# validation_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
# validation_sampler = SequentialSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
#
# test_data = TensorDataset(test_inputs, test_masks, test_labels)
# test_sampler = RandomSampler(test_data)
# test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
#
# print(test_inputs.shape)
# # import pickle
# # with open('./data/test_dataloader.pkl','wb') as f:
# #   pickle.dump(test_dataloader, f)
#
# # print('train_data:', len(train_labels))
# # print('valid_data:', len(valid_labels))
# # print('test_data:', len(test_labels))
# #
# # print(tokenizer.decode([2]))
#
# # print(train_inputs[0])
# # print(train_masks[0])
# #
# # print('훈련 데이터의 크기:', len(train_labels))
# # print('검증 데이터의 크기:', len(data_Preprocess2.validation_labels))
# # print('테스트 데이터의 크기:', len(test_labels))