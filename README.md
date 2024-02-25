### Benchmarking BERT-based NLP models on Embedded Devices (Raspberry Pi, Jetson, UDOO and UP board)

_Artifact and code for "Processing Natural Language on Embedded Devices: How Well Do Modern Models Perform?", ICPE'24._  

**How to use the code?**  


We used different embedded platform for conducting Intent Classification (IC) and Name Entity Recognition (NER) tasks.
We have trained the Bert model offline and put it on the BERT folder inside Intent Classification or NER folder. Saved model can be downloaded from the following link given below.  

Install the necessary python library listed in `required python library`  

Update the file path and txt file path in `inference_script_intentc.py` (line 81,88) and `inference_script_ner.py` (line 24,68).

Saved model link:https://doi.org/10.6084/m9.figshare.23304692.v2  https://doi.org/10.6084/m9.figshare.23304662 (you might need to open an account in Fig Share )

To measure the energy we used UM25C energy meter.     
To measure system memory consumption, we used `@profile` method of python.   
Deatils can be found by analyzing the code of `inference_script_ner.py`      

We run our coad in different embedded platforms and general purpouse computer. Anyone can test the code in general purpouse computer.
**Data-set link**   


HuRic Dataset: https://github.com/crux82/huric      
Go Emotions Dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset        
WNUT'17 Dataset: https://github.com/leondz/emerging_entities_17         
CoNLL Dataset: https://ebanalyse.github.io/NERDA/datasets/     

![Screenshot-61](https://github.com/CPS2RL/NLP-on-Embedded-Devices/assets/71979845/2690e3a1-27af-4b9f-8523-6ec05df8df8a)

Fig: Utterance processing steps of a voice-controlled embedded device.

The arXiv preprint of our findings is available here: https://arxiv.org/abs/2304.11520      
