# BCFNet: A Balanced Collaborative Filtering Network with Attention Mechanism

This is our official implementation for the paper:

Zi-Yuan Hu, Jin Huang, Zhi-Hong Deng, Chang-Dong Wang,  Ling Huang,  Jian-Huang Lai, Philip S. Yu. [BCFNet: A Balanced Collaborative Filtering Network with Attention Mechanism](https://arxiv.org/abs/2103.06105) 

​       In this paper, as an extension of DeepCF, before feeding the vectors into DNNs, we first input them into a feed-forward attention layer which can improve the representation ability of the deep neural networks. By allowing different parts to contribute differently when compressing them to a single representation, attention-based architectures can learn to focus their “attention” to specific parts. Higher weights indicate that the corresponding factors are more informative for the recommendation. In addition, to alleviate the overfitting issue and offset the weakness of MLP in capturing low-rank relations, a balance module is introduced by means of generalized matrix factorization model (GMF). 

​        Therefore, a novel model named Balanced Collaborative Filtering Network (BCFNet) is proposed, which consists of three sub-models, namely attentive representation learning (BCFNet-rl), attentive matching function learning (BCFNetml) and balance module (BCFNet-bm).  

​        Different types of representation learning-based methods and matching function learning-based methods can be integrated under the BCFNet framework. 

## Environment Settings
We use Keras with tensorflow as the backend. 
- Keras version: '2.1.4'
- tensorflow-gpu version:  '1.7.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run CFNet-rl:
```
python DMF.py --dataset ml-1m --epochs 20
```

Run CFNet-ml:
```
python MLP.py --dataset ml-1m --epochs 20
```

Run CFNet (without pre-training): 
```
python CFNet.py --dataset ml-1m --epochs 20 --lr 0.01
```

Run CFNet (with pre-training):
```
python CFNet.py --dataset ml-1m --epochs 20 --lr 0.0001  --learner sgd  --dmf_pretrain Pretrain/ml-1m_DMF_XX.h5 --mlp_pretrain Pretrain/ml-1m_MLP_XX.h5
```

### Dataset
We provide all four processed datasets: MovieLens 1 Million (ml-1m), MovieLen 100k (ml100k), LastFm (lastfm), Amazon Toy (AToy) Amazon Music (AMusic),  Amazon Baby (ABaby), Amazon Beauty (ABeauty), filmtrust, 

**train_rating.csv**

- Train file.
- Each Line is a training instance: `userID\t itemID\t rating\t timestamp (if have)`

**test_rating.csv**

- Test file (positive instances). 
- Each Line is a testing instance: `userID\t itemID\t rating\t timestamp (if have)`

**test_negative.csv**

- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: `(userID,itemID)\t negativeItemID1\t negativeItemID2 ...`

## Citation
```
@article{bcfnet,
  title={BCFNet: A Balanced Collaborative Filtering Network with Attention Mechanism},
  author={Zi-Yuan Hu, Jin Huang, Zhi-Hong Deng, Chang-Dong Wang,  Ling Huang,  Jian-Huang Lai, Philip S. Yu},
  year={2021}
}
```
If the code helps you in your research, please also cite:
```
@misc{bcfnet,
  author =       {Zi-Yuan Hu, Jin Huang, Zhi-Hong Deng, Chang-Dong Wang,  Ling Huang,  Jian-Huang Lai, Philip S. Yu},
  title =        {BCFNet},
  howpublished = {\url{https://github.com/beilineili/BCFNet}},
  year =         {2021}
}
```
