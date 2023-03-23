import os, yaml

import pandas as pd
import numpy as np

from utils import *
from models.models import train_model, evaluate_model, calibrate_model
from models.model_utils import generate_random_curves
from einops import rearrange
from joblib import dump

from sklearn.metrics import roc_auc_score, recall_score, precision_score

import tensorflow as tf
# fix random seeds for reproc
seed = 12345
tf.keras.utils.set_random_seed(seed)

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main(config):

    print('')
    # GET PARAMS #####################################################
    params = get_params(config)
    model_run = 'newangles_run_nseq{:}_lseq{:}_batch{:}_lr{:}_sample{:}_reg{:}_units{:}_depth{:}_dense{:}_smooth{:}_heads{:}'.format( params['model_parameters']['num_sequences'],
                                                                            params['model_parameters']['sequence_length'],
                                                                            params['model_parameters']['batch_size'],
                                                                            params['model_parameters']['learning_rate'],
                                                                            params['model_parameters']['oversample'],
                                                                            params['model_parameters']['regularisation'],
                                                                            params['model_parameters']['units'],
                                                                            params['model_parameters']['depth'],
                                                                            params['model_parameters']['num_dense'],
                                                                            params['model_parameters']['label_smoothing'],
                                                                            params['model_parameters']['attention_heads'])
    if params['model_parameters']['sigmoid_attention']:
        model_run = model_run + '_sigmoid'
    if params['processing']['add_features']:
        model_run = model_run + '_extras'
    if params['model_parameters']['augmentation']:
        model_run = model_run + '_augment'
    if params['model_parameters']['residuals']:
        model_run = model_run + '_rb'
    if params['model_parameters']['squeeze']:
        model_run = model_run + '_se'
    if params['model_parameters']['meta_data']:
        meta = True
        model_run = model_run + '_' + '_'.join(params['model_parameters']['meta_data'])
    else:
        meta = False

    outdir = 'data/' + model_run + '/'
    checkpointdir = 'checkpoints/' + model_run + '/'
    restart = False
    if os.path.isdir(outdir) and os.path.isdir(checkpointdir):
        restart = True
        print('model already started - trying to restart')
        print('')
    else:
        os.makedirs(outdir, exist_ok = True)
        os.makedirs(checkpointdir, exist_ok = True)
        print('model directory: {:}'.format(outdir))
        print('')

        with open(outdir + 'config.yaml', 'w') as file:
            yaml.dump(params, file)

    # PARAMS #####################################################
    # data - location of processed trajectory data
    datadir = params['data']['datadir']

    # preprocessing options
    do_smooth_data = params['processing']['smooth']
    smooth_window = params['processing']['smooth_window']
    add_features = params['processing']['add_features']

    # cv
    num_folds = params['cross-validation']['cv']

    # model
    data_gen_params = get_datagen_params(params)
    model_params = get_model_params(params)
    training_params = get_training_params(params)

    # callbacks
    cb_params = params['callbacks']
    #####################################################################

    # LOAD DATA #########################################################
    data, info = load_data(datadir)
    info['label'] = (info['gma_vid_score'] != 'normal GM').astype(int) #1 is abnormal
    info['group'] = (info['group'] != 'term control').astype(int) #1 is preterm

    if restart:
        tmp = np.load(outdir + '{:}_data.npy'.format(info['video'].iloc[0]))
        data_gen_params['length'] = len(tmp)
        data_gen_params['num_comps'] = tmp.shape[1]

    else:
        # PROCESS DATA ######################################################
        info[['video', 'label']].to_csv(outdir + 'labels.csv', index=None)
        print('saving labels to {:}labels.csv: abormal = 1'.format(outdir))

        data, _ = process_data(data, add_features=add_features,
                            do_smooth_data=do_smooth_data,
                            smooth_window=smooth_window,
                            )
        # get number of keypoints/components
        num_comps = np.shape(data)[-1]
        data_gen_params['num_comps'] = num_comps

        # CREATE CLIPS AND SAVEOUT###########################################
        print('')
        print('saving to {:}'.format(outdir))

        for n,d in enumerate(tqdm(data)):
            vid_name = info['video'].iloc[n]
            outfile = outdir + '{:}_data.npy'.format(vid_name)
            strided = np.squeeze(np.lib.stride_tricks.sliding_window_view(d, (data_gen_params['seq_length'], num_comps)).copy())
            strided = rearrange(strided, 'windows time comps -> windows comps time')
            # reduce number of windows to save some space
            strided = strided[::int(params['processing']['decimate_factor'])]

            np.save(outfile, strided)

        # number of clips per subject
        data_gen_params['length'] = len(strided)

    # precompute random curves for data augmentation and save out for later
    curves = generate_random_curves(length = data_gen_params['seq_length'], sigma=1., max_knot=15)
    outfile = outdir + 'curves.npy'
    np.save(outfile, curves)
    print('saving random curves to {:}curves.npy'.format(outdir))

    # CROSS VALIDATION #####################################################
    vid_list = list(info['video'])
    labels = list(info['label'].values)
    # get meta data if specified
    if meta:
        metadata = list(info[params['model_parameters']['meta_data']].values)

    fold_losses = []
    for n in np.arange(num_folds):
        if os.path.exists(checkpointdir + 'FOLD{:04d}/calibrator.joblib'.format(n)):
            print('FOLD {:} already run ***********************************'.format(n))
        else:
            # get train/test split
            train_fold_index, test_fold_index = get_train_test_split(info, split=.15, random_state=n*seed) #fix random state for reproc
            print('FOLD {:} *****************************************************'.format(n))
            # output folders
            folddir = outdir + 'FOLD{:04d}/'.format(n)
            chkfolddir = checkpointdir + 'FOLD{:04d}/'.format(n)
            os.makedirs(folddir, exist_ok=True)
            os.makedirs(chkfolddir, exist_ok=True)

            # split data into folds
            train_x = [vid_list[i] for i in train_fold_index]
            train_y = [labels[i] for i in train_fold_index]

            test_x = [vid_list[i] for i in test_fold_index]
            test_y = [labels[i] for i in test_fold_index]

            train_a = [metadata[i] for i in train_fold_index] if meta else None
            test_a = [metadata[i] for i in test_fold_index] if meta else None

            print('FOLD')
            print('number of train: {:}'.format(len(train_x)))
            print('number of positive train: {:}'.format(np.sum(train_y)))
            print('number of test: {:}'.format(len(test_x)))
            print('number of positive test: {:}'.format(np.sum(test_y)))

            # save out fold ids to fold dir
            np.savetxt(folddir + 'train_ids.txt', train_x, delimiter=',', fmt='%s')
            np.savetxt(folddir + 'test_ids.txt', test_x, delimiter=',', fmt='%s')

            # train
            model, loss, generators = train_model(train_x, train_y, train_a, info.iloc[train_fold_index],
                                        outdir, chkfolddir,
                                        data_gen_params, model_params, cb_params, training_params,
                                        split = .12, random_state=n)

            # calibrate
            calibrator = calibrate_model(generators['validation'], model)
            dump(calibrator, chkfolddir + 'calibrator.joblib')

            # predict test data
            all_predictions, all_calibrated_predictions = evaluate_model(test_x, test_y, test_a, model, calibrator, outdir, data_gen_params)
            print('original scores:-')
            print('AUC score: {:04f}'.format(roc_auc_score(test_y, all_predictions)))
            print('recall score: {:04f}'.format(recall_score(test_y, np.array(all_predictions)>.5)))
            print('precision score: {:04f}'.format(precision_score(test_y, np.array(all_predictions)>.5)))
            print('')

            print('calibrated scores:-')
            print('AUC score: {:04f}'.format(roc_auc_score(test_y, all_calibrated_predictions)))
            print('recall score: {:04f}'.format(recall_score(test_y, np.array(all_calibrated_predictions)>.5)))
            print('precision score: {:04f}'.format(precision_score(test_y, np.array(all_calibrated_predictions)>.5)))
            print('')

            pred_out = pd.DataFrame((test_x, test_y, all_predictions, all_calibrated_predictions)).T
            pred_out.columns = (['id', 'label', 'prediction', 'calibrated_predictions'])
            pred_out.to_csv(folddir + 'validation_predictions.csv', index=None)
            loss.to_csv(folddir + 'losses.csv', index=None)


if __name__ == '__main__':
    import argparse
    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise NameError(string)

    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('-c', '--config', type=file_path, required=True,
                help='configuration file')

    args = parser.parse_args()

    main(args.config)
