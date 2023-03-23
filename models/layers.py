from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

class AttentionWithContext(Layer):
    # https://www.kaggle.com/sermakarevich/hierarchical-attention-network
    # https://www.cc.gatech.edu/~dyang888/docs/naacl16.pdf

    """
    Input shape
        3D tensor with shape: (samples, steps, features)
    Output shape
        2D tensor with shape: (samples, features)
    """

    def __init__(self, units=32, kernel_initializer='he_uniform',
                 W_regularizer=None, u_regularizer=None, bias_regularizer=None,
                 W_constraint=None, u_constraint=None, bias_constraint=None,
                 bias=True, aggregate=True, sigmoid=False, **kwargs):

        self.init = kernel_initializer
        self.units = units
        self.aggregate = aggregate
        self.sigmoid = sigmoid
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(bias_constraint)

        self.use_bias = bias

        super(AttentionWithContext, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W= self.add_weight(shape=(self.units, input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(self.units,),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)


    def call(self, x):

        uit = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.use_bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.squeeze(K.dot(uit, K.expand_dims(self.u)), axis=-1)

        a = K.exp(ait)

        if self.sigmoid:
            a = 1/(1 + K.exp(-ait))
            a = K.expand_dims(a)
            weighted_input = x * a
            aggregate_output = K.mean(weighted_input, axis=1)
        else: #softmax
            # in some cases especially in the early stages of training the sum may be almost zero
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            a = K.expand_dims(a)
            weighted_input = x * a
            aggregate_output = K.sum(weighted_input, axis=1)

        if self.aggregate:
            return a, aggregate_output
        else:
            return a, weighted_input

    def compute_output_shape(self, input_shape):
        if self.aggregate:
            return (input_shape[0], input_shape[-2], 1),(input_shape[0], input_shape[-1])
        else:
            return (input_shape[0], input_shape[-2], 1), input_shape


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'aggregate': self.aggregate,
            'W_regularizer': self.W_regularizer,
            'u_regularizer': self.u_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'u_constraint': self.u_constraint,
            'b_constraint': self.b_constraint,
            'use_bias': self.use_bias
        })
        return config
