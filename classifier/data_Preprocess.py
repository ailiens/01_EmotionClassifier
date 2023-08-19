import pandas as pd
import openpyxl
df = pd.read_csv('./data/finance_data.csv')



# # 중립:0 ,긍정:1, 부정:2로 변경
# df['labels'] =df['labels'].replace(['neutral', 'positive', 'negative'],[0,1,2])
# print(df['labels'])
# print(df.head())
#
# # 변경한 데이터 프레임 csv로 저장
# # df.to_csv('./data/finance_data_num.csv', index=False, encoding='utf-8-sig')
# df1 = pd.read_csv('./data/finance_data_num.csv')
# print(df.head())



# df2 = pd.read_excel('./data/감성대화말뭉치(최종데이터)_Training.xlsx')
# print(df2.head())
# 중립:0 ,긍정:1, 부정:2로 변경
# df2['labels'] =df2['labels'].replace(['neutral', 'positive', 'negative'],[0,1,2])
# print(df['labels'])
# print(df.head())

# 변경한 데이터 프레임 csv로 저장
# df.to_csv('./data/finance_data_num.csv', index=False, encoding='utf-8-sig')
# df2 = pd.read_csv('./data/emotion_corpus.csv')
# print(df.head())


