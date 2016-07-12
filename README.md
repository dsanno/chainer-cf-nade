# CF-NADE

Implementation of ["A Neural Autoregressive Approach to Collaborative Filtering"](http://arxiv.org/abs/1605.09477)

# Requirements

* [Chainer 1.10.0](http://chainer.org/)

# How to use

## Downaload and extract dataset

Download MovieLens 1M/10M dataset from http://grouplens.org/datasets/movielens/ and extract it.

## Convert dataset

```
$ python src/convert_dataset.py dataset/ml-1m/ratings.dat dataset/ml-1m/movies.dat dataset/ml-1m/ratings.pkl
```

For 10M dataset you have to specify minimum rating value and rating unit .

```
$ python src/convert_dataset.py -m 0.5 -u 0.5 dataset/ml-10M100K/ratings.dat dataset/ml-10M100K/movies.dat dataset/ml-10M100K/ratings.pkl
```

## Train

For MovieLens 1M dataset:

```
$ python src/train.py -g 0 -o model/test.model -d dataset/ml-1m/dataset.pkl -b 512 --lr 0.001
```

For MovieLens 10M dataset:

```
$ python src/train.py -g 0 -o model/test.model -d dataset/ml-1M100K/dataset.pkl -b 512 -e 50 --lr 0.0005
```

Optiions:

* -g (--gpu) `<GPU device index>`: Optional  
GPU device index. Negative number indicates using CPU (default: -1)
* -o (--output) `<File path>`: Required  
Output model file path
* -d (--data_file) `<File path>`: Required  
Dataset file path
* -b (--batch_size) `<int>`: Optional  
Mini batch size for training (default: 512)
* -l (--layer_num) `<int>`: Optional  
Number of neural network layers (default: 1)
* ordinal_weight `float`: Optional  
Ordinal loss function weight (default: 1)
* --iter `<int>`: Optional  
Iteration of training (default: 200)
* --save_iter `<int>`: Optional  
Iteration of saving model (default: 10)
* --lr `<float>`: Optional  
Learning rate: alpha of Adam (default: 1e-3)
* --lr_decay_iter `<int>`: Optional  
Iteration interval of learning rate decay. (default: 60)
* --lr_decay_ratio `<int>`: Optional  
Ratio of learning rate after decay. (default: 0.25)
* --weight_decay `float`: Optional  
Weight decay (default: 0.015)
* --random_seed `<int>`: Optional  
Random seed (default: 1)
* --item_base: Optional  
Do item-base prediction if set
