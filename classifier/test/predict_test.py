from transformers import BertTokenizer, BertForSequenceClassification
# from classifier.test.model_test import model
from transformers import pipeline
import torch



tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
num_labels = 11
model = BertForSequenceClassification.from_pretrained('Klue/bert-base', num_labels=num_labels)
# model.cuda()
# model.load_state_dict(torch.load('../model/Bert_11_emotions_model_kp.pt'))
model.load_state_dict(torch.load('c:/Users/tjoeun/Documents/TJE_2nd_proj/classifier/model/Bert_11_emotions_model_kp.pt'))

pipe = pipeline("text-classification", model=model.cuda(), tokenizer=tokenizer, device=0, max_length=512,
                return_all_scores=True, function_to_apply='softmax')

# result = pipe('SK하이닉스가 매출이 급성장하였다')
# print("감정분류 결과: ", result)
# print()

pipe = pipeline("text-classification", model=model.cuda(), tokenizer=tokenizer, device=0, max_length=512, function_to_apply='softmax')

label_dict = {'LABEL_0': '중립',
              'LABEL_1': '불안',
              'LABEL_2': '분노',
              'LABEL_3': '놀람',
              'LABEL_4': '상처',
              'LABEL_5': '당황',
              'LABEL_6': '기쁨',
              'LABEL_7': '행복',
              'LABEL_8': '혐오',
              'LABEL_9': '공포',
              'LABEL_10': '슬픔',
              }

def prediction(query):
    result = pipe(query)

    return [label_dict[result[0]['label']]]

# print(prediction(input('입력:')))
# print("감정분류 결과: ", prediction('패스트캠퍼스가 매출이 급성장하였다'))
# print("감정분류 결과: ", prediction('나는 천재다'))