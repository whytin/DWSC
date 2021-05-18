import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.initializers import VarianceScaling, glorot_normal
import metrics
import deep_means
from keras.datasets import mnist
import numpy as np
from sklearn import preprocessing
from keras.callbacks import Callback

init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
batch_size=2000
p_dim=8
output_act = 'linear'
dataset='fmnist'


if dataset == 'fmnist':
    recon_co=K.variable(1.0)
    lamb=K.variable(0.5)
    alpha=K.variable(0.0005)
    beta=K.variable(0.15)
if dataset == 'mnist':
    recon_co=K.variable(1.0)
    lamb=K.variable(1.0)
    alpha=K.variable(0.0005)
    beta=K.variable(0.15)
if dataset == 'mnist' or 'fmnist':
    n_clusters=10
    original_dims = 784

if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)

elif dataset == 'fmnist':
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)



class EpochBegin(Callback):
    def __init__(self, lamb, alpha, beta, recon_co, dataset, dmeans_model):
        self.lamb = lamb 
        self.alpha = alpha 
        self.beta = beta
        self.recon_co = recon_co
        self.dataset = dataset
        self.dmeans_model = dmeans_model

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % 10 == 0:
            print("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))

    def on_epoch_end(self, epoch, logs={}):
        epochBegin(epoch, self.dmeans_model)




def epochBegin(epoch, dmeans_model):
    if (epoch+1) % 10 == 0:
        _, l1, l2, l3, w = dmeans_model.predict(x, batch_size=2000)
        y_pred = w.argmax(axis=1)
        acc = metrics.acc(y, y_pred)
        nmi = metrics.nmi(y, y_pred)
        ari = metrics.ari(y, y_pred)
        if acc>best_result['acc']:
           best_result['acc'] = acc
           best_result['nmi'] = nmi
           best_result['ari'] = ari
        print('ACC:', acc)
        print('NMI:', nmi)
        print('ARI:', ari)

def train():
    global best_result
    best_result = {'acc':0, 'nmi':0, 'ari':0}
    dmeans = deep_means.DeepMeans(dims=[original_dims,500,500,2000,10], cluster_dims=[100,n_clusters], n_clusters=n_clusters,lamb=lamb, alpha=alpha, beta=beta, batch_size=batch_size, output_act=output_act, p_dim=p_dim, init=init)
    dmeans_model = dmeans.build()
    optimizer = Adam(lr=0.02, epsilon=10e-8)
    dmeans_model.compile(optimizer='adam', loss=['mse', lambda y_ture, y_pred: y_pred, lambda y_ture, y_pred: y_pred, lambda y_ture, y_pred: y_pred, None], loss_weights={'decoder_0':recon_co, 'cluster_loss':lamb, 'sam_loss':alpha, 'con_loss':beta})
    init_subspace_weights = dmeans_model.get_layer(name='clustering').get_weights()
    if dataset=='fmnist':
        dmeans_model.load_weights('fm_ae_weight.h5')
        print('load weight successful.')
    elif dataset=='mnist':
        dmeans_model.load_weights('ae_weight.h5')
        print('load weight successful.')
    epoch_begin = EpochBegin(lamb,alpha, beta, recon_co, dataset, dmeans_model)
    dmeans_model.fit(x, [x, x, x, x], batch_size=batch_size, shuffle=True, epochs=300, callbacks=[epoch_begin])
    print('best result:', best_result)

if __name__ == '__main__':
    train()
