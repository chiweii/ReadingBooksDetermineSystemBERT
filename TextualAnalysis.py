# Author : Felix Chi
# Date : 2020/09

import os
import sys
import json
import pandas as pd
import tensorflow as tf

os.chdir("/var/www/html/analysis/project/bert-master");

#分析測試(未知)樣本類別，並將其放到test_output中 注意：這裡的init_checkpoint要指定到已訓練的模型

os.environ['BERT_Chinese_DIR'] = 'chinese_L-12_H-768_A-12';
os.environ['Demo_DIR'] = 'input/textual';
os.environ['TRAINED_CLASSIFIER'] = 'Demo_output_new';

os.system('python run_classifier.py \
  --task_name=demo \
  --do_predict=true \
  --data_dir=$Demo_DIR \
  --vocab_file=$BERT_Chinese_DIR/vocab.txt \
  --bert_config_file=$BERT_Chinese_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=test_output_textual')

#利用pandas套件讀入測試樣本(未知樣本)在各類別(成績)的機率值

df=pd.read_csv('test_output_textual/test_results.tsv',sep='\t',header=None)


#利用pandas套件中的idxmax找出每篇測試(未知)文章最大的機率值，並利用此機率值來決定其類別(預測分數)

preLabels=df.idxmax(axis=1)
preLabels.to_csv('test_output_textual/preLabels_new.tsv',index=False)

# print(df.head())
# print(preLabels.head())
result = {};
result['accuracy'] = str(df.values);
result['recommand'] = str(preLabels.values);
return_result = json.dumps(result); 
print(return_result);