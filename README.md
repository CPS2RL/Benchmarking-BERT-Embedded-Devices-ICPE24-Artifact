### Benchmarking BERT-based NLP models on Embedded Devices (Raspberry Pi)

**How to use the code?**  


We used different embedded platform for conducting Intent Classification (IC) and Name Entity Recognition (NER) tasks.
We have trained the Bert model offline and put it on the BERT folder inside Intent Classification or NER folder.
Install the necessary python library listed in `required python library`  

Update the file path and txt file path in `inference_script_intentc.py` and `inference_script_ner.py`.

Saved model link:https://figshare.com/account/home#/projects/169256

To measure the energy we used UM25C energy meter.     
To measure system memory consumption, we used `@profile` method of python.   
Deatils can be found by analyzing the code of `inference_script_NER.py`    
  
**Data-set link**   


HuRic Dataset: https://github.com/crux82/huric      
Go Emotions Dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset        
WNUT'17 Dataset: https://github.com/leondz/emerging_entities_17         
CoNLL Dataset: https://ebanalyse.github.io/NERDA/datasets/     

![Screenshot-61](https://github.com/CPS2RL/NLP-on-Embedded-Devices/assets/71979845/2690e3a1-27af-4b9f-8523-6ec05df8df8a)

Fig: Utterance processing steps of a voice-controlled embedded device.

The arXiv preprint of our findings is available here: https://arxiv.org/abs/2304.11520      
