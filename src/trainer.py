import numpy as np
import six

from chainer import functions as F
from chainer import cuda
from chainer import Variable

from softmax_cross_entropy import weight_softmax_cross_entropy

def make_rating_matrix(x, r, item_num, rating_num):
    y = np.zeros((x.shape[0], item_num, rating_num), dtype=np.float32)
    for i in six.moves.range(x.shape[0]):
        index = x[i] >= 0
        y[i, x[i, index], r[i, index]] = 1
    r_to_v = np.tri(rating_num, dtype=np.float32)
    y = y.dot(r_to_v)
    return y.reshape((x.shape[0], -1))

def make_target(x, r, item_num):
    y = np.full((x.shape[0], item_num), -1, dtype=np.int32)
    for i in six.moves.range(x.shape[0]):
        index = x[i] >= 0
        y[i, x[i, index]] = r[i, index]
    return y

def loss_mask(t, rating_num):
    b, n = t.shape
    flat_t = t.ravel()
    mask = np.zeros((b * n, rating_num), dtype=np.float32)
    mask[six.moves.range(b * n), flat_t] = 1
    mask[flat_t < 0,:] = 0
    return mask.reshape((b, n, rating_num)).transpose((0, 2, 1))

def ordinal_loss(y, mask):
    xp = cuda.get_array_module(y.data)
    volatile = y.volatile
    b, c, n = y.data.shape
    max_y = F.broadcast_to(F.max(y, axis=1, keepdims=True), y.data.shape)
    y = y - max_y
    sum_y = F.broadcast_to(F.expand_dims(F.sum(y, axis=1), 1), y.data.shape)
    down_tri = np.tri(c, dtype=np.float32)
    up_tri = down_tri.T
    w1 = Variable(xp.asarray(down_tri.reshape(c, c, 1, 1)), volatile=volatile)
    w2 = Variable(xp.asarray(up_tri.reshape(c, c, 1, 1)), volatile=volatile)
    h = F.exp(F.expand_dims(y, -1))
    h1 = F.convolution_2d(h, w1)
    h1 = F.convolution_2d(F.log(h1), w1)
    h2 = F.convolution_2d(h, w2)
    h2 = F.convolution_2d(F.log(h2), w2)
    h = F.reshape(h1 + h2, (b, c, n))
    return F.sum((h - sum_y - y) * mask) / b

class CfNadeTrainer(object):

    def __init__(self, net, optimizer, epoch_num=100, batch_size=512, device_id=-1, ordinal_weight=1, rating_unit=1):
        self.net = net
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device_id = device_id
        self.ordinal_weight = ordinal_weight
        self.rating_unit = rating_unit
        if device_id >= 0:
            self.xp = cuda.cupy
            self.net.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, x, r, valid_x, valid_r, test_x, test_r, callback=None):
        if self.device_id >= 0:
            with cuda.cupy.cuda.Device(self.device_id):
                return self.__fit(x, r, valid_x, valid_r, test_x, test_r, callback)
        else:
            return self.__fit(x, r, valid_x, valid_r, test_x, test_r, callback)

    def __fit(self, x, r, valid_x, valid_r, test_x, test_r, callback):
        batch_size = self.batch_size
        train_length = len(x)
        r_size = np.sum(x >= 0, axis=1).astype(np.int32)
        valid_r_size = np.sum(valid_x >= 0, axis=1).astype(np.int32)
        test_r_size = np.sum(test_x >= 0, axis=1).astype(np.int32)
        r_max_width = np.max(r_size)
        batch_x1 = np.full((batch_size, r_max_width - 1), -1, dtype=np.int32)
        batch_r1 = np.full((batch_size, r_max_width - 1), -1, dtype=np.int32)
        batch_x2 = np.full((batch_size, r_max_width - 1), -1, dtype=np.int32)
        batch_r2 = np.full((batch_size, r_max_width - 1), -1, dtype=np.int32)

        for epoch in six.moves.range(1, self.epoch_num + 1):
            data_order = np.random.permutation(train_length)
            train_loss = 0
            train_acc = 0
            train_valid_num = 0
            for i in six.moves.range(0, train_length, batch_size):
                batch_index = data_order[i:i + batch_size]
                batch_x1[:,:] = -1
                batch_r1[:,:] = -1
                batch_x2[:,:] = -1
                batch_r2[:,:] = -1
                batch_valid_num = 0
                train_size = min(batch_size, train_length - i)
                valid_size = np.zeros((train_size,), dtype=np.float32)
                for j in six.moves.range(train_size):
                    k = data_order[i + j]
                    size = r_size[k]
                    item_order = np.random.permutation(size)
                    train_num = np.int32(np.floor(np.random.rand() * (size - 1)) + 1)
                    valid_num = size - train_num
                    batch_x1[j,:train_num] = x[k,item_order[:train_num]]
                    batch_r1[j,:train_num] = r[k,item_order[:train_num]]
                    batch_x2[j,:valid_num] = x[k,item_order[train_num:]]
                    batch_r2[j,:valid_num] = r[k,item_order[train_num:]]
                    batch_valid_num += valid_num
                    valid_size[j] = valid_num
                self.net.zerograds()
                batch_x = make_rating_matrix(batch_x1[:train_size,:], batch_r1[:train_size,:], self.net.item_num, self.net.rating_num)
                batch_t = make_target(batch_x2[:train_size,:], batch_r2[:train_size,:], self.net.item_num)
                weight = r_size[data_order[i:i + train_size]].astype(np.float32) / (valid_size + 1e-6)
                loss, acc = self.__forward(batch_x, batch_t, weight)
                loss.backward()
                self.optimizer.update()
                train_loss += float(loss.data) * batch_valid_num
                train_acc += float(acc.data)
                train_valid_num += batch_valid_num

            valid_loss = 0
            valid_acc = 0
            valid_valid_num = 0
            for i in six.moves.range(0, train_length, batch_size):
                batch_valid_num = np.sum(valid_r[i:i + batch_size] >= 0)
                batch_x = make_rating_matrix(x[i:i + batch_size], r[i:i + batch_size], self.net.item_num, self.net.rating_num)
                batch_t = make_target(valid_x[i:i + batch_size], valid_r[i:i + batch_size], self.net.item_num)
                batch_r_size = r_size[i:i + batch_size].astype(np.float32)
                batch_valid_r_size = valid_r_size[i:i + batch_size].astype(np.float32)
                weight = (batch_r_size + batch_valid_r_size) / (batch_valid_r_size + 1e-6)
                loss, acc = self.__forward(batch_x, batch_t, weight, train=False)
                valid_loss += float(loss.data) * batch_valid_num
                valid_acc += float(acc.data)
                valid_valid_num += batch_valid_num

            test_loss = 0
            test_acc = 0
            test_valid_num = 0
            for i in six.moves.range(0, train_length, batch_size):
                batch_valid_num = np.sum(test_r[i:i + batch_size] >= 0)
                batch_x = make_rating_matrix(x[i:i + batch_size], r[i:i + batch_size], self.net.item_num, self.net.rating_num)
                batch_t = make_target(test_x[i:i + batch_size], test_r[i:i + batch_size], self.net.item_num)
                batch_r_size = r_size[i:i + batch_size].astype(np.float32)
                batch_test_r_size = test_r_size[i:i + batch_size].astype(np.float32)
                weight = (batch_r_size + batch_test_r_size) / (batch_test_r_size + 1e-6)
                loss, acc = self.__forward(batch_x, batch_t, weight, train=False)
                test_loss += float(loss.data) * batch_valid_num
                test_acc += float(acc.data)
                test_valid_num += batch_valid_num

            callback(epoch, train_loss / train_valid_num, (train_acc / train_valid_num) ** 0.5, valid_loss / valid_valid_num, (valid_acc / valid_valid_num) ** 0.5, test_loss / test_valid_num, (test_acc / test_valid_num) ** 0.5)

    def __forward(self, batch_x, batch_t, weight, train=True):
        xp = self.xp
        x = Variable(xp.asarray(batch_x), volatile=not train)
        t = Variable(xp.asarray(batch_t), volatile=not train)
        y = self.net(x, train=train)

        b, c, n = y.data.shape
        mask = Variable(xp.asarray(np.broadcast_to(weight.reshape(-1, 1, 1), (b, c, n)) * loss_mask(batch_t, self.net.rating_num)), volatile=not train)
        if self.ordinal_weight == 0:
            loss = F.sum(-F.log_softmax(y) * mask) / b
        elif self.ordinal_weight == 1:
            loss = ordinal_loss(y, mask)
        else:
            loss = (1 - self.ordinal_weight) * F.sum(-F.log_softmax(y) * mask) / b + self.ordinal_weight * ordinal_loss(y, mask)

        acc = self.__accuracy(y, t)
        return loss, acc

    def __accuracy(self, y, t):
        xp = self.xp
        b, c, n = y.data.shape
        v = np.arange(c, dtype=np.float32).reshape((1, -1, 1)).repeat(b, axis=0).repeat(n, axis=2)
        v = Variable(xp.asarray(v), volatile=True)
        r = F.sum(v * F.softmax(Variable(y.data, volatile=True)), axis=1)
        c = Variable(t.data >= 0, volatile=True)
        t = Variable(t.data.astype(np.float32), volatile=True)
        r = F.where(c, r, t)
        return F.sum(((r - t) * self.rating_unit) ** 2)
