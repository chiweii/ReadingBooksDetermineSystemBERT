from bert_serving.client import BertClient
#from google.colab import drive
import sys
# drive.mount('/content/drive')

import os

# os.system('ls')

import pandas as pd

os.chdir("/var/www/html/analysis/project/bert-master")
dfAll=pd.read_csv('input/All.csv',sep='\t',header=None,quoting=3)
dfAll.head()

from sklearn.model_selection import train_test_split 

dfAll.shape

dfTrain, dfTest, yTrain, yTest=train_test_split(dfAll,dfAll[0],train_size=0.7,stratify=dfAll[0])

dfTrain.head()

dfTrain.to_csv('input/train.tsv',sep='\t',index=False,header=None)
dfTrain.to_csv('input/dev.tsv',sep='\t',index=False,header=None)

dfTest.head()

dfTest.to_csv('input/test.tsv',sep='\t',index=False,header=None)

# import tensorflow

import tensorflow as tf

# print(tf.VERSION)

# tensorflow_version

"""利用訓練樣本訓練模型(這範例只能分類)"""

import os
os.environ['BERT_Chinese_DIR'] = 'chinese_L-12_H-768_A-12'
os.environ['Demo_DIR'] = 'input'

os.system('python run_classifier.py \
  --task_name=demo \
  --do_train=true \
  --do_eval=true \
  --data_dir=$Demo_DIR \
  --vocab_file=$BERT_Chinese_DIR/vocab.txt \
  --bert_config_file=$BERT_Chinese_DIR/bert_config.json \
  --init_checkpoint=$BERT_Chinese_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=50.0 \
  --output_dir=Demo_output_new')

"""分析測試(未知)樣本類別，並將其放到test_output中

注意：這裡的init_checkpoint要指定到上面細訓練的模型
"""

import os
os.environ['BERT_Chinese_DIR'] = 'chinese_L-12_H-768_A-12'
os.environ['Demo_DIR'] = 'input'
os.environ['TRAINED_CLASSIFIER'] = 'Demo_output_new'

os.system('python run_classifier.py \
  --task_name=demo \
  --do_predict=true \
  --data_dir=$Demo_DIR \
  --vocab_file=$BERT_Chinese_DIR/vocab.txt \
  --bert_config_file=$BERT_Chinese_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=test_output')

"""利用pandas套件讀入測試樣本(未知樣本)在各類別(成績)的機率值"""

df=pd.read_csv('test_output/test_results.tsv',sep='\t',header=None)
df.head()

"""利用pandas套件中的idxmax找出每篇測試(未知)文章最大的機率值，並利用此機率值來決定其類別(預測分數)"""

preLabels=df.idxmax(axis=1)
preLabels.to_csv('test_output/preLabels_new.tsv',index=False)
preLabels.head()

""" 
以下是在訓練模型時，要比較預測分數與專家分數時，觀察模型在測試樣本預測能力時才需要用到。 

讀入測試樣本的專家評分
"""

expLabels=dfTest[0]
expLabels.head()

"""利用sklearn套件來得到混淆矩陣，比較不同分數下，專家判定與模型判定的差異。"""

from sklearn import metrics

expLabels.shape
preLabels.shape

"""(樣本數,參數)"""

mcs = metrics.confusion_matrix(expLabels,preLabels)
print(mcs)

"""計算測試樣本正確率"""

mas = metrics.accuracy_score(expLabels,preLabels)
print(mas)
mfs = metrics.f1_score(expLabels,preLabels,average='macro')
print(mfs)