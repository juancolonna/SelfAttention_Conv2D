import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SelfAttention_Conv2D(keras.layers.Layer):
    def __init__(self, filters=None, num_heads=None, gamma=0.01, activation='linear', return_attention_scores=False, trainable=True):
        super().__init__(trainable=trainable)
        self.gamma = tf.Variable(initial_value=gamma, trainable=True, name='gamma')
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None
        self.num_heads = num_heads
        self.c = filters
        self.activation = activation
        self.scores = return_attention_scores

    def build(self, input_shape):

        if self.c is None:
            self.c = input_shape[-1]

        if self.num_heads is None:
            self.num_heads = input_shape[-1]

        self.f = self.block(self.num_heads) # [n, w*h, heads]
        self.g = self.block(self.num_heads) # [n, w*h, heads]
        self.h = self.block(self.num_heads) # [n, w*h, heads]

        # output feature maps -> [n, w, h, filters]
        self.v = layers.Conv2D(self.c, 1, 1, activation=self.activation)

    @staticmethod
    def block(c):
        return keras.Sequential([
            layers.Conv2D(c, 1, 1),   # [n, w, h, heads] 1*1conv
            layers.Reshape((-1, c)),  # [n, w*h, heads]
            ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)    # [n, w*h, heads]
        g = self.g(inputs)    # [n, w*h, heads]
        h = self.h(inputs)    # [n, w*h, heads]
        s = tf.matmul(f, g, transpose_b=True)   # [n, w*h, heads] @ [n, heads, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, heads] = [n, w*h, heads]
        d = inputs.shape        # [n, w, h, channels]
        cs = context_wh.shape   # [n, w*h, heads]
        context = self.gamma * tf.reshape(context_wh, [-1, d[1], d[2], cs[-1]])    # [n, w, h, heads]
        o = self.v(context) + inputs   # residual -> [n, w, h, filters]

        if self.scores:
            return o, context
        else:
            return o
