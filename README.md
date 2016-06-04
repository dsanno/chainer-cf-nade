# CF-NADE

Implementation of ["A Neural Autoregressive Approach to Collaborative Filtering"](http://arxiv.org/abs/1605.09477)

# Requirements

* [Chainer](http://chainer.org/)

# How to use

## Downaload and extract dataset

Download MovieLens 1M/10M dataset from http://grouplens.org/datasets/movielens/ and extract it.

## Convert dataset

```
$ python src/convert_dataset.py dataset/ml-1m/ratings.dat dataset/ml-1m/ratings.pkl
```

For 10M dataset you have to specify minimum rating value and rating unit .

```
$ python src/convert_dataset.py -m 0.5 -u 0.5 dataset/ml-10m/ratings.dat dataset/ml-10m/ratings.pkl
```

## Train

```
$ python src\train.py -g -1 -o model\test.model -d dataset\ml-1m\dataset.pkl -b 512 --lr 0.0001
```

Optiions:

* -g (--gpu) <GPU device index>: Optional  
GPU device index. Negative number indicates using CPU (default: -1)
* -o (--output) <File path>: Required
Output model file path
* -d (--data_file) <File path>: Required
Dataset file path
* -b (--batch_size) <int>: Optional
Mini batch size for training (default: 512)
* --iter <int>: Optional
Iteration of training (default: 100)
* --lr <float>: Optional
Learning rate: alpha of Adam (default: 1e-4)
* --random_seed <int>: Optional
Random seed (default: 1)
