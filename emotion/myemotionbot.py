from transformers import BertTokenizer, BertForSequenceClassification
# from classifier.test.model_test import model
from transformers import pipeline
import torch
from classifier.db.Database import Database
from classifier.db.DatabaseConfig import *
from classifier.test.predict_test import prediction
# 모델
# tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
# num_labels = 11
# model = BertForSequenceClassification.from_pretrained('Klue/bert-base', num_labels=num_labels)
# # model.cuda()
# model.load_state_dict(torch.load('../clssifier/model/Bert_11_emotions_model_kp.pt'))

def getMessage(query):
    try:
        db = Database(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
        )
        db.connect()
        # 의도파악
        predict = prediction(query)
        # print(query)
        # print("predict=", predict)
        # intent_name = intent.labels[intent_predict]
        # # 개체명 파악
        # ner_predicts = ner.predict(query)
        # ner_tags = ner.predict_tags(query)
        # 답변 검색
        # try:
        #     f = FindAnswer(db)
        #     answer_text, answer_image = f.search(intent_name, ner_tags)
        #     answer = f.tag_to_word(ner_predicts, answer_text)
        # except:
        #     answer = "죄송합니다. 질문 내용을 이해하지 못했습니다."
        #     answer_image = None
        json = {
            "Query": query,
            # "Answer": answer,
            # "AnswerImageUrl": answer_image,
            "predict": predict,
            # "NER": str(ner_predicts)
        }
        return json
    except Exception as ex:
        print(ex)


# if __name__ == '__main__':
#     # msg = getMessage('짜장면 두그릇 주문합니다.')
#     msg = getMessage(input('내용 입력:'))
#     print(msg)
