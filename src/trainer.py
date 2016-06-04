import numpy as np
import six

from chainer import functions as F
from chainer import cuda
from chainer import Variable

class CfNadeTrainer(object):

    def __init__(self, net, optimizer, epoch_num=100, batch_size=512, device_id=-1):
        self.net = net
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device_id = device_id
        if device_id >= 0:
            self.xp = cuda.cupy
            self.net.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, x, r, test_x, test_r, callback=None):
        if self.device_id >= 0:
            with cuda.cupy.cuda.Device(self.device_id):
                return self.__fit(x, r, test_x, test_r, callback)
        else:
            return self.__fit(x, r, test_x, test_r, callback)

    def __fit(self, x, r, test_x, test_r, callback):
        batch_size = self.batch_size
        train_length = len(x)
        r_size = np.sum(x >= 0, axis=1).astype(np.int32)
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
                self.net.zerograds()
                loss, acc = self.__forward(batch_x1[:train_size,:], batch_r1[:train_size,:], batch_x2[:train_size,:], batch_r2[:train_size,:])
                loss.backward()
                self.optimizer.update()
                train_loss += float(loss.data) * batch_valid_num
                train_acc += float(acc.data) * batch_valid_num
                train_valid_num += batch_valid_num
            callback(epoch, train_loss / train_valid_num, train_acc / train_valid_num, None, None)

    def __forward(self, batch_x1, batch_r1, batch_x2, batch_r2, train=True):
        xp = self.xp
        x1 = Variable(xp.asarray(batch_x1), volatile=not train)
        r1 = Variable(xp.asarray(batch_r1), volatile=not train)
        x2 = Variable(xp.asarray(batch_x2), volatile=not train)
        r2 = Variable(xp.asarray(batch_r2), volatile=not train)
        y = self.net(x1, r2, x2, train=train)
        # set use_cudnn False to avoid 'cupy.cudnn supports c-contiguous arrays only' Error
        loss = F.softmax_cross_entropy(y, r2, use_cudnn=False)
        acc = self.__accuracy(y, r2)
#        print float(acc.data), float(F.accuracy(y, r2).data)
        return loss, acc

    def __accuracy(self, y, t):
        xp = self.xp
        r = xp.argmax(y.data, axis=1).astype(np.float32)
        t = Variable(t.data.astype(np.float32), volatile=True)
        r = xp.where(t.data < 0, r, t.data)
        r = Variable(r, volatile=True)
        return F.mean_squared_error(r, t)
