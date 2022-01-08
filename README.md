# MiceRec
MiceRec code for SDM 2022

Our experimental code is implemented in Python 3.6 using TensorFlow 1.13.1.

Datasets detail is follow paper ComiRec: Controllable Multi-Interest Framework for Recommendation  (https://github.com/THUDM/ComiRec)

http://jmcauley.ucsd.edu/data/amazon/ 

https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1


You can run model like:
python src/train.py --dataset book --model_type MiceRec-SA --learning_rate 0.001 --weight_decay 0.9 --dis_loss_weight 0.05 --neg_num 10 --num_interest 8

ps:
Due to the time conflict of the company's patent application, 
the details of the model part are temporarily invisible. 
We will add all the codes after the patent application is successful. 
It is expected to be March 2022.
Thanks for your patience.
