import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers as I

class NadeIn(chainer.Chain):

    def __init__(self, item_num, hidden_num, rating_num, encoding_size=-1):
        wscale = np.sqrt(0.1)
        if encoding_size <= 0:
            super(NadeIn, self).__init__(
                a=L.Linear(item_num * rating_num, hidden_num, initialW=I.Uniform(0.06), initial_bias=I.Constant(0)),
            )
        else:
            super(NadeIn, self).__init__(
                a=L.Linear(item_num * rating_num, encoding_size, nobias=True, initialW=I.Uniform(0.06)),
                b=L.Linear(encoding_size, hidden_num, initialW=I.Uniform(0.06), initial_bias=I.Constant(0)),
            )

    def __call__(self, x1, train=True):
        """
        in_type:
            x1: float32
        in_shape:
            x1: (batch_size, train_item_num * rating_num)
        out_type: float32
        out_shape: (batch_size, hidden_num)
        """

        xp = cuda.get_array_module(x1.data)
        h = self.a(x1)
        if hasattr(self, 'b'):
            h = self.b(h)
#        h = F.dropout(h, train=train)
        return F.tanh(h)

class NadeHidden(chainer.Chain):

    def __init__(self, hidden_num):
        wscale = np.sqrt(0.1)
        super(NadeHidden, self).__init__(
            w=L.Linear(hidden_num, hidden_num, initialW=I.Uniform(0.06), initial_bias=I.Constant(0)),
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
#        h = F.dropout(h, train=train)
        return F.tanh(h)

class NadeOut(chainer.Chain):

    def __init__(self, item_num, hidden_num, rating_num, encoding_size=-1):
        wscale = np.sqrt(0.1)
        if encoding_size <= 0:
            super(NadeOut, self).__init__(
                p = L.Linear(hidden_num, rating_num * item_num, initialW=I.Uniform(0.06), initial_bias=I.Constant(0)),
            )
        else:
            super(NadeOut, self).__init__(
                p = L.Linear(hidden_num, encoding_size, nobias=True, initialW=I.Uniform(0.06)),
                q = L.Linear(encoding_size, rating_num * item_num, initialW=I.Uniform(0.06), initial_bias=I.Constant(0)),
            )
        self.item_num = item_num
        self.rating_num = rating_num

    def __call__(self, h, train=True):
        """
        in_type:
            h: float32
        in_shape:
            h: (batch_size, hidden_num)
        out_type: float32
        out_shape: (batch_size, rating_num, predicted_item_num)
        """

        xp = cuda.get_array_module(h.data)
        h = self.p(h)
        if hasattr(self, 'q'):
            h = self.q(h)
        h = F.reshape(h, (-1, self.rating_num, self.item_num, 1))
        w = chainer.Variable(xp.asarray(np.tri(self.rating_num, dtype=np.float32).reshape(self.rating_num, self.rating_num, 1, 1)), volatile=h.volatile)
        h = F.convolution_2d(h, w)
        return F.reshape(h, (-1, self.rating_num, self.item_num))

class CfNade(chainer.Chain):

    def __init__(self, item_num, layer_num=1, hidden_num=500, rating_num=5, encoding_size=-1):
        super(CfNade, self).__init__(
            l_in=NadeIn(item_num, hidden_num, rating_num, encoding_size),
            l_out=NadeOut(item_num, hidden_num, rating_num, encoding_size),
        )
        self.hidden_links = []
        for i in six.moves.range(layer_num - 1):
            link = NadeHidden(hidden_num)
            self.add_link('h_{}'.format(i + 1), link)
            self.hidden_links.append(link)
        self.item_num = item_num
        self.rating_num = rating_num

    def __call__(self, x, train=True):
        """

        Args:
            x (Variable): observed items.
                type: int32
                shape: (batch_size, train_item_num * rating_num))
            train (bool): True for train, otherwise False.
        Returns:
            Variable:
            type: float32
            shape: (batch_size, rating_num, predicted_item_num)
        """
        h = self.l_in(x, train=train)
        for link in self.hidden_links:
            h = link(h, train=train)
        h = self.l_out(h, train=train)
        return h
