from transformers.models.bert import BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from classifier.BertTokenizer import train_dataloader, train_data
import datetime
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from torch.optim import AdamW

import time
import random
import numpy as np
from tqdm import tqdm

# GPU setting
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print('cpu')

# Torch 버전
# print(torch.__version__)
# # CUDA 사용 가능여부(True, False)
# print(torch.cuda.is_available())
# # CUDA 버전
# print(torch.version.cuda)
# # GPU 사용가능 개수
# print(torch.cuda.device_count())
# # GPU 이름
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.current_device())


# 모델 로드
num_labels = 3
epochs = 2
batch_size=32
# batch_size=2

model = BertForSequenceClassification.from_pretrained('Klue/bert-base', num_labels=num_labels)
model.cuda()
# 옵티마이저 선택
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

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

# 랜덤 시드값.
seed_val = 777
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model.zero_grad()
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in tqdm(enumerate(train_dataloader)):
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping if it is over a threshold
        optimizer.step()
        scheduler.step()

        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))