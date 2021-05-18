from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf

class DeepMeans():
    def __init__(self, dims, cluster_dims, n_clusters, lamb, alpha,beta, batch_size,output_act, p_dim=1, act='relu', init='glorot_normal'):
        self.dims = dims
        self.cluster_dims = cluster_dims
        self.act = act
        self.init = init
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.p_dim = p_dim
        self.alpha = alpha
        self.output_act = output_act
        self.beta = beta
        self.lamb = lamb 
    def kronecker(self, list):
        z = K.reshape(list[0], [self.batch_size,1,self.dims[-1]])
        w = K.reshape(list[1], [self.batch_size,self.n_clusters,1])
        wz = K.reshape(w*z, [self.batch_size, self.n_clusters, self.dims[-1]])
        #wz = z*w
        return wz

    def cluster_loss(self, args):
        z, u, w = args
        z = K.tile(z, [self.n_clusters,1])
        z = K.reshape(z, [self.n_clusters, self.batch_size, self.dims[-1]])
        w = K.tile(w, [self.dims[-1], 1])
        w = K.reshape(w, [self.dims[-1], self.batch_size, self.n_clusters])
        w = tf.transpose(w, [2,1,0])
        loss = K.mean(w*K.square(z-u))

        return loss

    def s_transpose(self, s):
        s = tf.transpose(s, [1,0,2])
        s = K.reshape(s, [self.batch_size, self.n_clusters*self.dims[-1]])
        return s



    def diff(self, args):
        z, u =args
        u = K.tile(u, [self.batch_size, 1])
        u = K.reshape(u, [self.batch_size, self.n_clusters, self.dims[-1]])
        z = K.tile(z, [1, self.n_clusters])
        z = K.reshape(z, [self.batch_size, self.n_clusters, self.dims[-1]])
        diff = K.reshape(z-u, [self.batch_size, self.n_clusters*self.dims[-1]])
        return diff

    def w_loss(self, arg):
        return (1/self.batch_size)*(-tf.linalg.trace(tf.matmul(arg, tf.transpose(arg))))

    def target_distribution(self, q):
        weight = K.square(q) / K.sum(q, axis=0)
        p = K.transpose(K.transpose(weight) / K.sum(weight, axis=1))
        return self.kullback_leibler_divergence(p, q)

    def kullback_leibler_divergence(self, y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    def cov_loss(self, x):
        mean_x = tf.reduce_mean(x, axis=1, keep_dims=True)
        mx = tf.matmul(mean_x, tf.transpose(mean_x))
        vx = tf.matmul( x,tf.transpose(x)) / tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        cov_loss = -K.mean(K.square(cov_xx))
        return cov_loss


    def condition_loss(self, arg):
        average = K.mean(arg, axis=0)
        return tf.norm(average)


    def build(self):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(self.dims) - 1
        input_img = Input(batch_shape=(self.batch_size,self.dims[0]), name='input')
        x = input_img
        for i in range(n_stacks - 1):
            x = Dense(self.dims[i + 1], activation=self.act, kernel_initializer=self.init, name='encoder_%d' % i)(x)

        encoded = Dense(self.dims[-1], kernel_initializer=self.init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

        x = encoded

        for i in range(n_stacks - 1, 0, -1):
            x = Dense(self.dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(x)


        x = Dense(self.dims[0], kernel_initializer=self.init, activation=self.output_act, name='decoder_0')(x)
        decoded = x

        ssT, s= SoftAssignment(n_clusters=self.n_clusters, batch_size=self.batch_size, p_dim=self.p_dim, name='clustering')(encoded)


        w = encoded
        n_w = len(self.cluster_dims)-1

        for i in range(n_w):
            w = Dense(self.cluster_dims[i], activation=self.act, kernel_initializer=self.init, name='generw_%d' % i)(w)
        w = Dense(self.cluster_dims[-1], activation='softmax', kernel_initializer=self.init, name='generw_softmax')(w)


        cluster_loss = Lambda(lambda x: self.cluster_loss(x), name='cluster_loss')([encoded, ssT, w])
        w_loss = Lambda(lambda x: self.w_loss(x), name='sam_loss')(w)

        con_loss = Lambda(lambda x: self.condition_loss(x), name='con_loss')(w)
        return Model(inputs=input_img, output=[decoded, cluster_loss, w_loss, con_loss, w])
class Fuzzy(Layer):
    def __init__(self,n_clusters, batch_size, weights=None, **kwargs):
        super(Fuzzy, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.initial_weights=weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) ==2
        self.input_dim = input_shape[-1]
        self.input_spec=InputSpec(dtype=K.floatx(), shape=(self.batch_size, self.input_dim))
        self.fuzzy_w = self.add_weight((self.batch_size, self.n_clusters), initializer='glorot_normal', name='fuzzy_w')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.build = True
    def kronecker(self, list):
        z = K.reshape(list[0], [self.batch_size,1,self.input_dim])
        w = K.reshape(list[1], [self.batch_size,self.n_clusters,1])
        wz = K.reshape(w*z, [self.batch_size, self.n_clusters, self.input_dim])
        return wz
    def call(self,inputs):
        w = K.softmax(self.fuzzy_w, axis=1)
        fuzzy_z = self.kronecker([inputs, w])
        return [fuzzy_z, w]

    def compute_output_shape(self,input_shape):
        assert input_shape and len(input_shape)==2
        output_shape1=[self.batch_size, self.n_clusters, self.input_dim]
        output_shape2=[self.batch_size, self.n_clusters]
        return [tuple(output_shape1), tuple(output_shape2)]
    def get_config(self):
        config = {'n_clusters': self.n_clusters,
                  'batch_size': self.batch_size}
        base_config = super(Fuzzy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftAssignment(Layer):
    def __init__(self, n_clusters, batch_size, p_dim=1, weights=None, **kwargs):
        super(SoftAssignment, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.initial_weights = weights
        self.p_dim=p_dim
        self.input_spec = InputSpec(ndim=2)

    def fro_norm(self, w):
        return K.sqrt(K.sum(K.square(K.abs(w))))

    def cust_reg(self, w):
        m = tf.matmul(w, tf.transpose(w, [0,2,1])) - tf.eye(self.p_dim,self.p_dim)
        return self.fro_norm(m)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_dim = input_shape[-1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(self.batch_size, self.input_dim))
        self.S = self.add_weight((self.n_clusters, self.p_dim, self.input_dim), initializer='glorot_normal', name='S', regularizer=self.cust_reg)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True



    def call(self, inputs):
        SST= tf.matmul(tf.transpose(self.S, [0,2,1]), self.S)
        encode = K.reshape(K.tile(inputs, [self.n_clusters, 1]), [self.n_clusters, self.batch_size, self.input_dim])
        new_encode = tf.matmul(encode,SST)

        s_encode = tf.matmul(encode, tf.transpose(self.S, [0,2,1]))
        s_encode = tf.transpose(s_encode,[1,0,2])
        s_encode = tf.reshape(s_encode, [self.batch_size, self.n_clusters*self.p_dim])
        return [new_encode, s_encode]


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape)==2
        output_shape=[self.n_clusters, self.batch_size, self.input_dim]
       

        s_shape=[self.batch_size, self.p_dim*self.n_clusters]

        return [tuple(output_shape),tuple(s_shape)]

    def get_config(self):
        config = {'n_clusters': self.n_clusters,
                  'batch_size': self.batch_size,
                  'p_dim': self.p_dim}
        base_config = super(SoftAssignment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






