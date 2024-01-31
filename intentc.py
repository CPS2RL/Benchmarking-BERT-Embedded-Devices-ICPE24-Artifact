
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from model import tokenization_with_bert, BertClassifier
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertModel, BertConfig
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers.modeling_utils import PreTrainedModel
import time

from memory_profiler import profile
import requests
import psutil


def tokenization_with_bert(tokenizer: BertTokenizer, data: list, max_len: int):
    input_ids, attention_masks = list(), list()
    for sent in data:
        encoded_sent = tokenizer.encode_plus(text=sent, add_special_tokens=True, max_length=max_len,
                                             pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

class BertClassifier(PreTrainedModel):
    def __init__(self, h_size_bert, h_size_classifier, number_labels, config, freeze_bert=False, model_path=None):
        super(BertClassifier, self).__init__(config)
        self.h_size_bert = h_size_bert
        self.h_size_classifier = h_size_classifier
        self.number_labels = number_labels
        if model_path != None:
            self.bert = pickle.load(open("assets/saved_models/transformer/bert.pkl", "rb"))
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased', config=config,ignore_mismatched_sizes=True)

        self.classifier = nn.Sequential(
                                        nn.Linear(self.h_size_bert, self.h_size_classifier),
                                        nn.ReLU(), 
                                        nn.Linear(self.h_size_classifier, self.number_labels)
                                        )
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_h_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_h_state_cls)
        return logits
        
@profile
def fa(text):
    TEST_INPUTS, TEST_MASK = tokenization_with_bert(bert_tokenizer, [text], 64)
    model.eval()
    logits = model(TEST_INPUTS, TEST_MASK)
    pred_labels= torch.sigmoid(logits)
    pred_labels = pred_labels.detach().cpu().numpy()
    # preds = [(pred > 0.5) for pred in pred_labels ]
    # preds = np.asarray(preds)
    preds = (np.array(pred_labels) > 0.5).astype(int)
    new_preds = preds.reshape(1,-1).astype(int)
    pred_tags = mlb.inverse_transform(new_preds)
    # predict_labels = actions[np.argmax(new_preds.detach().numpy())]
    return pred_tags
        
mlb=MultiLabelBinarizer()
actions=["admiration", "amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
labels=mlb.fit_transform([actions])
text="list airlines flying from seattle to salt lake city"
path="/home/pi/ros2_foxy/src/intentc/intentc/bert/Layer8_90/"
config = AutoConfig.from_pretrained(path)
model = BertClassifier(h_size_bert=768, h_size_classifier=50, number_labels=28,config= config)
model.load_state_dict(torch.load(path+'pytorch_model.bin',map_location=torch.device('cpu')))
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset_path='/home/pi/ros2_foxy/src/intentc/intentc/test.txt'

t1=time.time()
for i in range(20):
	x=2
t1=(time.time()-t1)/20
arr=[]
with open(dataset_path) as f:
    lines = f.readlines()
    for line in lines:
        t2=time.time()
        for i in range(20):
            count=0
            pred_tags=fa(line)       
        t2=(time.time()-t2)/20
        arr.append(t2-t1)
        print(t2-t1)
        print(pred_tags)                
 
print(arr)            

