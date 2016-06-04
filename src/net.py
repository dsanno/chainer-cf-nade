import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

class NadeIn(chainer.Chain):

    def __init__(self, item_num, hidden_num, rating_num, encoding_num):
        super(NadeIn, self).__init__(
            a=L.EmbedID(item_num * rating_num, encoding_num, ignore_label=-1),
            b=L.Linear(encoding_num, hidden_num)
        )
        self.rating_num = rating_num

    def __call__(self, x1, r, train=True):
        """
        in_type:
            x1: int32
            r1: int32
        in_shape:
            x1: (batch_size, train_item_num)
            r1: (batch_size, train_item_num)
        out_type: float32
        out_shape: (batch_size, hidden_num)
        """

        xp = cuda.get_array_module(x1.data)
        condition = chainer.Variable(x1.data >= 0, volatile=x1.volatile)
        z = chainer.Variable(xp.zeros_like(x1.data), volatile=x1.volatile)
        h = F.where(condition, x1, z) * self.rating_num + r
        h = self.a(h)
        h = F.sum(h, axis=1)
        h = self.b(h)
        h = F.dropout(h, train=train)
        return F.tanh(h)

class NadeHidden(chainer.Chain):

    def __init__(self, hidden_num):
        super(NadeHidden, self).__init__(
            w=L.Linear(hidden_num, hidden_num),
        )

    def __call__(self, h, train=True):
        """
        in_type:
            h: float32
        in_shape:
            h: (batch_size, hidden_num)
        out_type: float32
        out_shape: (batch_size, hidden_num)
        """
        h = self.w(h)
        h = F.dropout(h, train=train)
        return F.tanh(h)

class NadeOut(chainer.Chain):

    def __init__(self, item_num, hidden_num, rating_num, encoding_num):
        super(NadeOut, self).__init__(
            p=L.EmbedID(item_num, rating_num * encoding_num, ignore_label=-1),
            b=L.EmbedID(item_num, rating_num, ignore_label=-1),
            q=L.Linear(hidden_num, encoding_num),
        )
        self.encoding_num = encoding_num

    def __call__(self, h, x2, train=True):
        """
        in_type:
            h: float32
            x2: int32
        in_shape:
            h: (batch_size, hidden_num)
            x2: (batch_size, predicted_item_num)
        out_type: float32
        out_shape: (batch_size, rating_num, predicted_item_num)
        """

        batch_size, item_num = x2.data.shape
        p = F.reshape(self.p(x2), (batch_size, -1, self.encoding_num))
        b = F.reshape(self.b(x2), (batch_size, -1))

        h = self.q(h)
        h = F.batch_matmul(p, h) + F.expand_dims(b, 2)
        return F.transpose(F.reshape(h, (batch_size, item_num, -1)), (0, 2, 1))

class CfNade(chainer.Chain):

    def __init__(self, item_num, layer_num=1, hidden_num=500, rating_num=5, encoding_num=50):
        super(CfNade, self).__init__(
            l_in=NadeIn(item_num, hidden_num, rating_num, encoding_num),
            l_out=NadeOut(item_num, hidden_num, rating_num, encoding_num),
        )
        self.hidden_links = []
        for i in six.moves.range(layer_num - 1):
            link = NadeHidden(hidden_num)
            self.add_link('h_{}'.format(i + 1), link)
            self.hidden_links.append(link)

    def __call__(self, x1, r, x2, train=True):
        """

        Args:
            x1 (Variable): observed items.
                type: int32
                shape: (batch_size, train_item_num)
            r (Variable): observed ratings.
                type: int32
                shape: (batch_size, train_item_num)
            x2 (Variable): predicted items.
                type: int32
                shape: (batch_size, predicted_item_num)
            train (bool): True for train, otherwise False.
        Returns:
            Variable:
            type: float32
            shape: (batch_size, rating_num, predicted_item_num)
        """
        h = self.l_in(x1, r, train=train)
        for link in self.hidden_links:
            h = link(h, train=train)
        h = self.l_out(h, x2, train=train)
        return h

def ordinary_softmax_cross_entropy(x):
    pass
