import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertForSequenceClassification
import pickle
import datetime
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss

# GPU setting
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def metrics(predictions, labels):
    y_pred = predictions
    y_true = labels

    # 사용 가능한 메트릭들을 사용한다.
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)

    # 메트릭 결과에 대해서 리턴
    metrics = {'accuracy': accuracy,
               'f1_macro': f1_macro_average,
               'f1_micro': f1_micro_average,
               'f1_weighted': f1_weighted_average}
    return metrics

# Load the test_dataloader from the file
with open('c:/Users/tjoeun/Documents/Classifier/classifier/data/test_dataloader.pkl', 'rb') as file:
    test_dataloader = pickle.load(file)

# 데이터 평가
num_labels = 11
model = BertForSequenceClassification.from_pretrained('Klue/bert-base', num_labels=num_labels)
model.cuda()
model.load_state_dict(torch.load('../model/Bert_11_emotions_model_kp.pt'))
# model.load_state_dict(torch.load('../model/0816.pt'))

t0 = time.time()
model.eval()
accum_logits, accum_label_ids = [], []

for step, batch in tqdm(enumerate(test_dataloader)):
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    for b in logits:
        accum_logits.append(np.argmax(b))

    for b in label_ids:
        accum_label_ids.append(b)

accum_logits = np.array(accum_logits)
accum_label_ids = np.array(accum_label_ids)
results = metrics(accum_logits, accum_label_ids)

print("Accuracy: {0:.4f}".format(results['accuracy']))
print("F1 (Macro) Score: {0:.4f}".format(results['f1_macro']))
print("F1 (Micro) Score: {0:.4f}".format(results['f1_micro']))
print("F1 (Weighted) Score: {0:.4f}".format(results['f1_weighted']))





### 됨
# t0 = time.time()
# model.eval()
# accum_logits, accum_label_ids = [], []
# #
# for step, batch in tqdm(enumerate(test_dataloader)):
#     b_input_ids, b_input_mask, b_labels = batch
#     b_input_ids = b_input_ids.to('cuda')
#     b_input_mask = b_input_mask.to('cuda')
#     b_labels = b_labels.to('cuda')
#
#     if step % 100 == 0 and not step == 0:
#         elapsed = format_time(time.time() - t0)
#         # elapsed = ft.format_time()
#         print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))
#
#     # for t in batch:
#     #     X,y=t
#     # b_input_ids, b_input_mask, b_labels = batch
#     with torch.no_grad():
#         outputs = model(b_input_ids,
#                         token_type_ids=None,
#                         attention_mask=b_input_mask)
#
#     logits = outputs[0]
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
#
#     for b in logits:
#         accum_logits.append(np.argmax(b))
#
#     for b in label_ids:
#         accum_label_ids.append(b)
#
# accum_logits = np.array(accum_logits)
# accum_label_ids = np.array(accum_label_ids)
# results = metrics(accum_logits, accum_label_ids)
#
# print("Accuracy: {0:.4f}".format(results['accuracy']))
# print("F1 (Macro) Score: {0:.4f}".format(results['f1_macro']))
# print("F1 (Micro) Score: {0:.4f}".format(results['f1_micro']))
# print("F1 (Weighted) Score: {0:.4f}".format(results['f1_weighted']))
