import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from einops import rearrange
from scipy.interpolate import CubicSpline
from scipy.stats import zscore
import tensorflow.keras.metrics as tfm

class Precision(tfm.Precision):
    'precision from logits'
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)

class Recall(tfm.Recall):
    'recall from logits'
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)

def generate_random_curves(length=128, num_curves=10000, sigma=0.2, min_knot=3, max_knot=15):
    # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
    """generate random cubic splines with random knot spacing along timeseries"""
    xx = []
    yy = []
    for c in np.arange(num_curves):
        knots = np.random.randint(min_knot, max_knot)
        xx.append((np.ones((1, 1))*(np.arange(0, length, (length-1)/(knots+1)))).transpose())
        yy.append(np.random.normal(loc=1.0, scale=sigma, size=(knots+2, 1)))

    x_range = np.arange(length)
    curves = np.array([CubicSpline(xx[i].reshape(-1), yy[i].reshape(-1))(x_range) for i in np.arange(num_curves)])

    return curves

class SequenceGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, data_path, meta=None,
                 to_fit=True, batch_size=1, num_comps=20, length=4500, oversample=1,
                 num_seq = 100, seq_length = 125,
                 weights=(1.0, 1.0), shuffle=True, validation=False, augmentation=False):
        """Initialization
        :param list_IDs: list of all video ids to use in the generator
        :param labels: list of labels (0/1)
        :param data_path: path to data location
        :param meta: optional list of metadata for each subject, may be None
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration (ignored in validation)
        :param num_comps: number of components or kerpoints in data
        :param length: number of clips in each video sample (should be the same for all)
        :param oversample: factor to oversample minority class (ignored in validation)
        :param num_seq: number of clips to extract per video during training (ignored in validation)
        :param seq_length: length of clips to extract
        :param weights: class weights
        :param shuffle: True to shuffle label indexes after every epoch
        :param validation: if validation, output class weights for loss
        :param augmentation: perform data aug. during training
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.data_path = data_path
        self.meta = meta
        if self.meta:
            self.num_meta = np.shape(self.meta)[1]
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.num_comps = num_comps
        self.length = length
        self.oversample = oversample
        self.num_seq = num_seq
        self.seq_length = seq_length
        self.weights = weights
        self.shuffle = shuffle
        self.validation = validation
        self.augmentation = augmentation
        if self.validation:
            self.num_samples = len(self.labels)
            self.oversample = 1
            self.batch_size = 1
            self.shuffle=False
        else:
            self.num_samples = np.sum([1-l for l in self.labels]) + np.sum(self.labels) * self.oversample
            self.curves = np.load(self.data_path + '/curves.npy') # load precomputed curves

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        if self.validation:
            list_IDs_repeat = self.list_IDs
            labels_IDs_repeat = self.labels
            if self.meta:
                meta_IDs_repeat = self.meta
        else:
            # oversample minority class
            repeats = [r * (self.oversample-1) + 1 for r in self.labels]
            list_IDs_repeat = np.repeat(self.list_IDs, repeats, axis=0)
            labels_IDs_repeat = np.repeat(self.labels, repeats, axis=0)
            if self.meta:
                meta_IDs_repeat = np.repeat(self.meta, repeats, axis=0)

        list_IDs_temp = [list_IDs_repeat[k] for k in indexes]
        label_IDs_temp = [labels_IDs_repeat[k] for k in indexes]
        meta_IDs_temp = [meta_IDs_repeat[k] for k in indexes] if self.meta else None

        # Generate data
        X = self._generate_X(list_IDs_temp, meta=meta_IDs_temp)

        if self.to_fit:
            y = np.array(label_IDs_temp, dtype=float)[:,np.newaxis]
            if self.validation:
                # get weights
                w = y.copy()
                w[y==0] = self.weights[0]
                w[y==1] = self.weights[1]
                return (X, y, w)
            else:
                return (X, y)
        else:
            return ([X])

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getsize__(self):
        """Return shape of X"""
        if self.validation:
            num_sequences = self.length
        else:
            num_sequences = self.num_seq

        return ((num_sequences, self.num_comps, self.seq_length), (1, self.num_meta)) if self.meta else (num_sequences, self.num_comps, self.seq_length)


    def _generate_X(self, list_IDs_temp, meta=None):
        """Generates data containing graph data
        :param list_IDs_temp: list of label ids to load
        :return: batch of graph sequences
        """

        X = []
        a = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X.append(self._load_clip_data('{:}/{:}_data.npy'.format(self.data_path, ID)))
            if meta:
                a.append(np.expand_dims(meta[i],0))

        return (np.array(X), np.array(a)) if meta else np.array(X)


    def _load_clip_data(self, tmp_data_path):
        """Load timeseries data
        :param image_path: path to image to load
        """
        strided = np.load(tmp_data_path, mmap_mode='r')

        if self.validation:
            # take every window
            x = np.array(strided)

        else:
            # random selection of clips
            rand = np.random.choice(len(strided), replace=False, size=self.num_seq)
            x = np.array(strided[rand])

            if self.augmentation:

                # random scale timeseries
                x_flat = rearrange(x, 'windows comps time -> (windows comps) time')
                x_flat = self._magnitude_warp(x_flat, 1)
                x = rearrange(x_flat, '(windows comps) time -> windows comps time', comps=self.num_comps)

                # random timewarps
                x = np.array([self._time_warp(x[s]) for s in np.arange(self.num_seq)])

        return x

    def _get_random_curves(self, num_curves):
        # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
        """select random cubic splines from precomputed examples"""
        rand_idx = np.random.choice(np.arange(self.curves.shape[0]), replace=False, size=num_curves)
        curves = np.array([self.curves[i,:] for i in rand_idx])

        return curves

    def _magnitude_warp(self, x, num_curves):
        # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
        """apply smooth magnitude scaling along timeseries"""
        curves = self._get_random_curves(num_curves)
        mn = np.mean(curves, axis=1)[:,np.newaxis]
        curves = np.exp((curves - mn) / (np.ptp(curves, axis=1)[:,np.newaxis]))

        return x * curves

    def _time_warp(self, x):
        # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
        """apply windowed time warping to whole clip (all timeseries together)"""
        tt = self._get_random_curves(1)
        tt = tt - np.min(tt)
        tt_cumul = np.cumsum(tt, axis=1)
        t_scale = (x.shape[1]-1)/tt_cumul[:,-1]
        tt_new = tt_cumul * t_scale
        x_range = np.arange(x.shape[1])

        return np.array([np.interp(x_range, tt_new.flatten(), x[i,:]) for i in range(len(x))])
