import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SelfAttention_Conv2D(keras.layers.Layer):
    def __init__(self, filters=None, gamma=0.01, activation='linear', return_attention_scores=False, trainable=True):
        super().__init__(trainable=trainable)
        self.gamma = tf.Variable(initial_value=gamma, trainable=True, name='gamma')
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None
        self.c = filters
        self.activation = activation
        self.scores = return_attention_scores

    def build(self, input_shape):
        if self.c is None:
            self.c = input_shape[-1]
        self.f = self.block(self.c)
        self.g = self.block(self.c)
        self.h = self.block(self.c)
        self.v = layers.Conv2D(self.c, 1, 1, activation=self.activation) # output feature maps

    @staticmethod
    def block(c):
        return keras.Sequential([
            layers.Conv2D(c, 1, 1),   # [n, w, h, c] 1*1conv
            layers.Reshape((-1, c)),  # [n, w*h, c]
            ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)    # [n, w*h, c]
        g = self.g(inputs)    # [n, w*h, c]
        h = self.h(inputs)    # [n, w*h, c]
        s = tf.matmul(f, g, transpose_b=True)   # [n, w*h, c] @ [n, c, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c] = [n, w*h, c]
        d = inputs.shape        # [n, w, h, c]
        cs = context_wh.shape   # [n, w*h, c]
        context = self.gamma * tf.reshape(context_wh, [-1, d[1], d[2], cs[-1]])    # [n, w, h, c]
        o = self.v(context) + inputs   # residual

        if self.scores:
            return o, self.attention
        else:
            return o