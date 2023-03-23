import glob

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, MaxPooling2D, BatchNormalization, Dense, Activation, Conv1D, Multiply, Add, GlobalAveragePooling2D

from einops.layers.keras import Rearrange, Reduce

from models.layers import AttentionWithContext
from utils import get_train_test_split
from models.model_utils import SequenceGenerator, Recall, Precision

from sklearn.linear_model import LogisticRegression

KERNEL_INIT = 'he_uniform'

def conv_block(x_in, units, kernel_size=3, padding='causal', activation='ReLU', residuals=False, squeeze_excite=True, reg=0.0):

    x = Conv1D(units,
                kernel_size = kernel_size,
                padding=padding,
                kernel_initializer=KERNEL_INIT,
                kernel_regularizer=tf.keras.regularizers.l2(l2=reg),
                use_bias=False)(x_in)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)

    if residuals:

        x = Conv1D(units,
                    kernel_size = kernel_size,
                    padding=padding,
                    kernel_initializer=KERNEL_INIT,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=reg),
                    use_bias=False)(x)

        if x_in.shape[-1] != units:
            x_in = Conv1D(units, kernel_size=1, kernel_initializer=KERNEL_INIT, kernel_regularizer=tf.keras.regularizers.l2(l2=reg))(x_in)

        x = Add()([x_in, x])
        x = Activation(activation)(x)
        x = BatchNormalization()(x)

    if squeeze_excite:
        x = se_block(x, units // 2, activation=activation, reg=reg)

    return x

def se_block(x, units, activation='ReLU', reg=0.0, return_weights=False):

    s = GlobalAveragePooling2D()(x)
    s = Dense(units, activation=activation, use_bias=False, kernel_initializer=KERNEL_INIT, kernel_regularizer=tf.keras.regularizers.l2(l2=reg))(s)
    s = Dense(x.shape[-1], activation='sigmoid', use_bias=False, kernel_initializer=KERNEL_INIT, kernel_regularizer=tf.keras.regularizers.l2(l2=reg))(s)

    if return_weights:
        return s, Multiply()([x, s])
    else:
        return Multiply()([x, s])

def get_model(input_shape,
              units = 8,
              kernel = 3,
              dropout = 0.0,
              activation = 'ReLU',
              depth=3,
              num_dense=2,
              return_attn=False,
              pool=2,
              reg=0.0001,
              residuals=False,
              squeeze_excite=True,
              attention_heads = 1,
              sigmoid = False):

    meta = True if len(input_shape) == 2 else False
    if meta:
        # send metadata to final layer via a 1D conv
        in1 = Input(shape=input_shape[0])
        in2 = Input(shape=input_shape[1])

        x_meta = BatchNormalization()(in2) # standardise
        x_meta = Conv1D(4, kernel_size=1, activation=activation, use_bias=False, kernel_initializer=KERNEL_INIT, kernel_regularizer=tf.keras.regularizers.l2(l2=reg))(x_meta)
        x_meta = tf.keras.layers.Flatten()(x_meta)

    else:
        in1 = Input(shape=input_shape)

    x = Rearrange('batch sample point time -> batch sample time point')(in1)

    # feature extraction
    for d in np.arange(depth):
        x = conv_block(x, units, kernel_size = kernel, padding='causal', activation=activation, reg=reg, residuals=residuals, squeeze_excite=squeeze_excite)
        x = MaxPooling2D(pool_size = (1, pool), strides = (1, pool))(x)

    # flatten
    x = Rearrange('batch sample time features -> batch sample (time features)')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    # fcn
    for _ in np.arange(num_dense):
        x = Dense(units, activation = activation, kernel_regularizer=tf.keras.regularizers.l2(l2=reg), kernel_initializer=KERNEL_INIT)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # sample attention
    if attention_heads > 1:
        all_a = []
        all_x = []
        for _ in np.arange(attention_heads):
            a_, x_ = AttentionWithContext(units=units, bias=True, kernel_initializer=KERNEL_INIT, W_regularizer=tf.keras.regularizers.l2(l2=reg), sigmoid=sigmoid)(x)
            all_x.append(x_)
            all_a.append(a_)

        a = tf.keras.layers.Average()(all_a)
        x = tf.keras.layers.Concatenate()(all_x)
    else:
        a, x = AttentionWithContext(units=units, bias=True, kernel_initializer=KERNEL_INIT,  W_regularizer=tf.keras.regularizers.l2(l2=reg), sigmoid=sigmoid)(x)

    #x = Reduce('batch sample features -> batch features', 'mean')(x)
    # add on meta data if available
    if meta:
        x = tf.keras.layers.Concatenate()([x, x_meta])

    if dropout > 0:
        x = Dropout(dropout)(x)
    # final prediction
    x =  tf.keras.layers.Dense(1, kernel_initializer=KERNEL_INIT, kernel_regularizer=tf.keras.regularizers.l2(l2=reg), activation=None)(x)

    if meta:
        if return_attn:
            model = tf.keras.Model([in1, in2], [x, a])
        else:
            model = tf.keras.Model([in1, in2], x)
    else:
        if return_attn:
            model = tf.keras.Model(in1, [x, a])
        else:
            model = tf.keras.Model(in1, x)

    return model

def train_model(x, y, a, info, datadir, checkdir, generator_params, model_params, callback_params, training_params, split=0.33, random_state=None):
    """train model returning model with best performance on inner validation split

    :param x, list of ids in training data
    :param y, list of labels for each id
    :param a, list of metadata (may be None)
    :param info, dataframe with info (id, video file name) for each subject
    :param datadir, path to data directory
    :param checkdir, path to save model checkpoints
    :param generator | model | callback_params, dict of parameters for generator, model, callbacks
    :param lr, learning rate
    :param label_smooth, amount of smoothing to apply to binary labels
    :param data_augmentation, flip, reverse and stretch samples
    :param proportion of data to test validation loss during training

    :return model, model with best performing weights
    :return loss_df, dataframe with training and val loss for each epoch"""

    # data generators
    train_index, test_index = get_train_test_split(info, split=split, random_state=random_state)
    train_x = [x[i] for i in train_index]
    train_y = [y[i] for i in train_index]
    val_x = [x[i] for i in test_index]
    val_y = [y[i] for i in test_index]

    train_a = [a[i] for i in train_index] if a else None
    val_a = [a[i] for i in test_index] if a else None


    print('MODEL VALIDATION')
    print('number of train: {:}'.format(len(train_x)))
    print('number of val: {:}'.format(len(val_x)))
    print('number of positive val: {:}'.format(np.sum(val_y)))

    # generators
    train_generator = SequenceGenerator(
                                train_x, train_y, datadir, meta=train_a,
                                **generator_params,
                                augmentation=training_params['data_augmentation'],
                                shuffle=True
                               )

    val_generator = SequenceGenerator(
                                val_x, val_y, datadir, meta=val_a,
                                **generator_params,
                                augmentation=False,
                                shuffle=False,
                                validation=True
                               )

    # compile model
    if a:
        _, point, feat = train_generator.__getsize__()[0]
        num_meta = train_generator.__getsize__()[1]

        model = get_model(
                                [(None, point, feat), num_meta],
                                **model_params
                                )
    else:
        _, point, feat = train_generator.__getsize__()
        model = get_model(
                                (None, point, feat),
                                **model_params
                                )

    # save plot of model
    tf.keras.utils.plot_model(model, to_file=datadir + '/model.png')
    print(model.summary())

    # metrics
    metrics = [ tf.keras.metrics.AUC(curve='ROC', name='auc', from_logits=True),
                tf.keras.metrics.AUC(curve='PR', name='pr', from_logits=True),
                Recall(name='recall', from_logits=True),
                Precision(name='precision', from_logits=True) ]

    # loss
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=training_params['label_smooth'], from_logits=True)

    # decaying learning rate schedule if final_lr =/= initial lr
    if training_params['final_lr'] < training_params['lr']:
        learning_rate_decay_factor = (training_params['final_lr'] / training_params['lr'])**(1 / training_params['num_epochs'])
        steps_per_epoch = train_generator.__len__()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=training_params['lr'],
                        decay_steps=steps_per_epoch,
                        decay_rate=learning_rate_decay_factor,
                        staircase=False)
    else:
        lr_schedule = training_params['lr']

    # callbacks
    cbs = [ tf.keras.callbacks.EarlyStopping(**callback_params, restore_best_weights=True, verbose=1)]
            #tf.keras.callbacks.TensorBoard(log_dir='logs3')]

    #compile
    model.compile(loss=loss, optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True), metrics = metrics)

    # fit
    # initial steps without early stopping
    print('initial training')
    history_pre = model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=250
                        )
    print('training with early stopping')
    history = model.fit(
                        train_generator,
                        validation_data=val_generator,
                        callbacks=cbs,
                        epochs=training_params['num_epochs']-250
                        )
    model.save(checkdir)

    # save loss curves
    train_loss = np.append(history_pre.history['loss'], history.history['loss'])
    val_loss = np.append(history_pre.history['val_loss'], history.history['val_loss'])
    loss_df = pd.DataFrame([train_loss, val_loss]).T
    loss_df.columns=['training_loss', 'validation_loss']

    return model, loss_df, {'training':train_generator, 'validation':val_generator}

def calibrate_model(generator, model):
    """Platt scaling: fits a logistic regression to model outputs to generate calibrated probs

    params: generator, data generator (not data used to train initial model)
    params: model, trained models

    return: calibrator, sklearn logistic regression model object"""
    print('')
    print('Calibrating')
    calibrator = LogisticRegression(penalty='none', class_weight='balanced', solver = 'lbfgs')

    assert generator.shuffle is False, 'validation data should not be shuffled'
    # get model probs
    proba = model.predict(generator)
    # make 0,1
    if model.loss.from_logits:
        proba = 1 / (1 + np.exp(-proba))
    # calibrate
    calibrator.fit(proba, generator.labels)

    # check correct direction (with poor model performance, occasionally predictions and probs are flipped)
    if calibrator.coef_ < 0:
        print('flipping')
        calibrator.coef_ = -calibrator.coef_
        calibrator.intercept_ = -calibrator.intercept_

    return calibrator


def evaluate_model(ids, labels, meta, model, calibrator, datadir, generator_params):
    """evaluates model performance on n_repeats random samples from each dataset in
    validaton data, returns each prediction and the average

    :param ids, list of ids in validation data
    :param labels, list of labels for each id
    :param meta, list of metadata, may be None
    :param model, trained model
    :param datadir, path to data directory
    :param generator_params, dict of parameters for generator
    :param n_repeats, how many random samples per subject

    :return val_predictions
    :return val_calibrated_predictions"""

    val_predictions = []
    calibrated_val_predictions = []

    print('')
    print('Validating')
    val_params = generator_params.copy()


    test_generator = SequenceGenerator(
                                        ids, labels, datadir, meta=meta,
                                        **val_params,
                                        validation=True,
                                        to_fit=False, shuffle=False
                                    )

    preds = model.predict(test_generator)
    if model.loss.from_logits:
        preds = 1 / (1 + np.exp(-preds))
    calibrated_preds = calibrator.predict_proba(preds)[:,1]

    val_predictions.append(np.array(preds))
    calibrated_val_predictions.append(np.array(calibrated_preds))

    return np.squeeze(np.array(val_predictions)), np.squeeze(np.array(calibrated_val_predictions))
