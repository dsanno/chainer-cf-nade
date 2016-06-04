import argparse
import cPickle as pickle
import glob
import numpy as np
import os
import six
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import serializers
from chainer import optimizers
from trainer import CfNadeTrainer
from net import CfNade

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CF-NADE')
    parser.add_argument('--gpu',           '-g', default=-1,    type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input',         '-i', default=None,  type=str, help='input model file path')
    parser.add_argument('--output',        '-o', required=True, type=str, help='output model file path')
    parser.add_argument('--data_file',     '-d', required=True, type=str, help='dataset file path')
    parser.add_argument('--batch_size',    '-b', default=512,   type=int, help='mini batch size')
    parser.add_argument('--iter',                default=200,   type=int, help='number of iteration')
    parser.add_argument('--lr',                  default=1e-4,  type=float, help='learning rate')
    parser.add_argument('--random_seed',         default=1,     type=int, help="random seed")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    device_id = args.gpu
    if device_id >= 0:
        cuda.check_cuda_available()

    with open(args.data_file, 'rb') as f:
        (user_num, item_num, train_data, test_data) = pickle.load(f)
    net = CfNade(item_num)
    optimizer = optimizers.Adam(args.lr)
    optimizer.setup(net)

    train_users, train_items, train_ratings, train_timestamps = train_data
    test_users, test_items, test_ratings, test_timestamps = test_data

    column_num = np.max(np.bincount(train_users))
    train_x = np.full((item_num, column_num), -1, dtype=np.int32)
    train_r = np.full((item_num, column_num), -1, dtype=np.int32)
    for i in six.moves.range(user_num):
        index = (train_users == i)
        length = np.sum(index)
        train_x[i,:length] = train_items[index]
        train_r[i,:length] = train_ratings[index]

    column_num = np.max(np.bincount(test_users))
    test_x = np.full((item_num, column_num), -1, dtype=np.int32)
    test_r = np.full((item_num, column_num), -1, dtype=np.int32)
    for i in six.moves.range(user_num):
        index = (test_users == i)
        length = np.sum(index)
        test_x[i,:length] = test_items[index]
        test_r[i,:length] = test_ratings[index]

    def progress_func(epoch, loss, accuracy, test_loss, test_accuracy):
        print 'epoch: {} done'.format(epoch)
        print('train mean loss={}, accuracy={}'.format(loss, accuracy))
        if test_loss is not None and test_accuracy is not None:
            print('test mean loss={}, accuracy={}'.format(test_loss, test_accuracy))
        if epoch % 10 == 0:
            serializers.save_npz(args.output, net)

    trainer = CfNadeTrainer(net, optimizer, args.iter, args.batch_size, device_id)
    trainer.fit(train_x, train_r, test_x, test_r, callback=progress_func)
