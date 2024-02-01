### Benchmarking BERT-based NLP models on Embedded Devices (Raspberry Pi)

**How to use the code?**  


We used different embedded platform for conducting Intent Classification (IC) and Name Entity Recognition (NER) tasks. Our implementation employed based on the pub-sub model of robot operation system (ROS). We have implemented the project on Raspberry Pi. Installation guide of ROS2 in Pi: https://docs.ros.org/en/foxy/How-To-Guides/Installing-on-Raspberry-Pi.html . You can find more information about the ROS pub-sub model in the provided link: https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html  
We have trained the Bert model offline and put it on the BERT folder inside Intent Classification or NER folder.  

Saved model link:https://figshare.com/account/home#/projects/169256

To measure the energy we used UM25C energy meter.     
To measure system memory consumption, we used @profile method of python.   
Deatils can be found by analyzing the code of `inference_script_NER.py`    
  
**Data-set link**   


HuRic Dataset: https://github.com/crux82/huric      
Go Emotions Dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset        
WNUT'17 Dataset: https://github.com/leondz/emerging_entities_17         
CoNLL Dataset: https://ebanalyse.github.io/NERDA/datasets/     

![Screenshot-61](https://github.com/CPS2RL/NLP-on-Embedded-Devices/assets/71979845/2690e3a1-27af-4b9f-8523-6ec05df8df8a)

Fig: Utterance processing steps of a voice-controlled embedded device.

The arXiv preprint of our findings is available here: https://arxiv.org/abs/2304.11520      
