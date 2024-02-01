# External imports
#from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
# External imports
import json
import sklearn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from model import tokenization_with_bert, BertClassifier
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import pickle
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from memory_profiler import profile
import requests
import psutil


## update the path with saved model
BERT_LOC='/home/pi/ros2_foxy/src/ner/ner/bert/Layer8_90/'


def load_model(BERT_LOC):
    tokenizer = AutoTokenizer.from_pretrained(BERT_LOC, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(BERT_LOC,ignore_mismatched_sizes=True)
    return {'tokenizer':tokenizer,'model':model}


def load_ner_model(model_name):
    if "bert" == model_name:
        return load_model(BERT_LOC)
    else:
        print("No NER technique detected.")
    return


def get_labels(text, ner):
    ner_model = pipeline('ner', model=ner["model"], tokenizer=ner["tokenizer"], grouped_entities=True)
    # Test the sequence
    return ner_model(text)



def annotate(text, active_feature, ner):
    label_sequence = get_labels(text, ner)
    return label_sequence
    

@profile
def annotate_text(text, active_feature, ner_name, ner):
    if "bert" == ner_name:
        return annotate(text, active_feature, ner)
    elif "roberta" == ner_name:
        return annotate(text, active_feature, ner)
    elif "xlnet" == ner_name:
        return annotate(text, active_feature, ner)
    elif "distilbert" == ner_name:
        return annotate(text, active_feature, ner)
    return text



##update the path with intentc_test.txt file
dataset_path='/home/pi/ros2_foxy/src/ner/ner/intentc_test.txt'

# with open(dataset_path, "r") as f:
#     data = json.loads(f.read())

model_name='bert'
model = load_ner_model(model_name)
c=0

t1=time.time()
for i in range(100):
    x=2
t1=(time.time()-t1)/100

arr=[]
with open(dataset_path) as f:
    lines = f.readlines()
    for line in lines:
    #line='Indiana is 13th in the Eastern Conference with a 25-55 record. At the trade deadline, the Pacers reset their competitive timeline by sending All-Star Domantas Sabonis, Justin Holiday, Jeremy Lamb, and a second-round pick in 2023 to the Sacramento Kings for Tyrese Haliburton, Buddy Hield, and Tristan Thompson.'
        t2=time.time()
        for i in range(100):  
            count=0
            feature_annotation = annotate_text(line, 'ORG', model_name, model)
            
        t2=(time.time()-t2)/100
        arr.append(t2-t1)
        print(t2-t1)
        #print(feature_annotation)
        
print(arr)
