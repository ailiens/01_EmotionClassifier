import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

# from datasets import load_dataset
#
# all_data = load_dataset(
#     "csv",
#     data_files = {
#         "train" : "./data/finance_data_num.csv"
#     },
# )
# # print(all_data)
#
# # dataset의 train_test_split으로 8:2로 분리하고 train, test로 저장하기
# cs = all_data['train'].train_test_split(0.2)
# train_cs = cs['train']
# test_cs = cs['test']
#
# # print(train_cs, '\t',test_cs)
#
# # valid 데이터를 위해 train_cs를 다시 분리
# cs = train_cs.train_test_split(0.2)
# train_cs = cs['train']
# valid_cs = cs['test']
#
# train_labels = train_cs['labels']
# validation_labels = valid_cs['labels']
# test_labels = test_cs['labels']
#
#
# train_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', train_cs['kor_sentence']))
# validation_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', valid_cs['kor_sentence']))
# test_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', test_cs['kor_sentence']))

############################
# 데이터 가져오기
df = pd.read_csv('./data/combined_data_num.csv')
# print(df.head())

# feature, label 나누기
sentences = df['sentence']
labels = df['labels']

train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
train_sentences, valid_sentences, train_labels, valid_labels = train_test_split(train_sentences, train_labels, test_size=0.1, random_state=42)

train_labels = train_labels.tolist()
valid_labels = valid_labels.tolist()
test_labels = test_labels.tolist()

# Add '[CLS]' and '[SEP]' to the sentences
train_sentences = ['[CLS] ' + str(sentence) + ' [SEP]' for sentence in train_sentences]
valid_sentences = ['[CLS] ' + str(sentence) + ' [SEP]' for sentence in valid_sentences]
test_sentences = ['[CLS] ' + str(sentence) + ' [SEP]' for sentence in test_sentences]

#pickle로 저장

# 실행 확인
# print(type(train_sentences)) # list
# print(type(train_labels))  # dataframe
# print(train_labels[:5])
## 결과 ##
# <class 'list'>
# <class 'list'>
# [0, 1, 0, 0, 2]
# torch.Size([970, 128])
#########


# print(train_sentences)
# print('*' * 50)
# print(validation_sentences)
# print('*' * 50)
# print(test_sentences)
# print('*' * 50)
# print(train_labels)
# print('*' * 50)
# print(validation_labels)
# print('*' * 50)
# print(test_labels)

# <class 'datasets.arrow_dataset.Dataset'> ##
# print(type(train_sentences))
# print(type(valid_sentences))
# print(type(test_sentences))
# # #
# print(type(train_labels))
# print(valid_labels)
# print(test_labels)



# print(train_cs, '\t', valid_cs, '\t', test_cs)
#
# print('샘플 문장 출력 : ', train_cs['kor_sentence'][1])
# print('샘플 레이블 출력: ', train_cs['labels'][1])
# print('샘플 문장 출력 : ', train_cs['kor_sentence'][0])
# print('샘플 레이블 출력: ', train_cs['labels'][0])


# print(type(train_cs['kor_sentence']))
# print(type(train_cs['labels']))
#
# print(type(train_cs))
#
# train_cs
# valid_cs
# test_cs import pickle
# original_list = train_cs['kor_sentence']
#
# new_list = original_list[:]
#
# 리스트를 파일로 저장하는 함수
train_data_list = train_sentences
valid_data_list = valid_sentences
test_data_list = test_sentences

data_list1 = train_data_list
data_list2 = valid_data_list
data_list3 = test_data_list

file_path = ('C:/Users/tjeun/Documents/TJE_2nd_proj/classifier/data')
def save_list_to_file(data_list2, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list2, file)
#
# # 파일에서 리스트를 불러오는 함수
# def load_list_from_file(file_path):
#     with open(file_path, 'rb') as file:
#         data_list = pickle.load(file)
#     return data_list
#
save_list_to_file(data_list2, './data/valid_data_loader.pkl')
#
# train_features = load_list_from_file('./data/train/train_features.pkl')
# print(train_features)


