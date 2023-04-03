# post-processing pipeline
# 
# Elyse Passmore 2021
# script for article and to process new video
# 
# To be run after DLC label predictions
# 1) quality control, ensure >70% of bodypoints labelled on average throughout video
# 2) adjust for camera movement, align torso midline to vertical
# 3) outlier removal
# 4) gap fill
# 5) normalisation (time and size)

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import glob
import pickle
import os
import cv2

import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from PIL import Image

def quality_control(points, cutoff, qc): 
    #points: dataframe of x y coordinates ouput from DLC
    #cutoff: cut off value for DLC confidence in prediction (likelihood), recommend 0.2
    #qc: Quality control value, % of points required for further analysis. recommend 70

    likelihood = points.loc[:,points.columns.get_level_values(2)=='likelihood']
    index = likelihood.le(cutoff).values
    
    points_clean = points.loc[:,np.logical_or(points.columns.get_level_values(2)=='x', 
                                                  points.columns.get_level_values(2)=='y')]
    points_clean.values[np.repeat(index,2, axis=1)] = np.nan

    #only need x values as just checking if labelled
    points_x = points_clean.loc[:,points_clean.columns.get_level_values(2)=='x']
    #number of video frames
    nframes = points_x.shape[0]
    
    #count number of nan values
    labelled_num = points_x.notna().sum(axis=0)
    labelled_perc = (labelled_num/nframes)*100
    labelled_perc.index = labelled_perc.index.get_level_values(1)
    labelled_total = np.average(labelled_perc.values)
    if labelled_total >= qc:
        include = True
        print('{:.0f}% of points labeled, video suitable for analysis'.format(labelled_total))
    else:
        include = False
        print('{:.0f}% of points labeled, video not suitable for anlaysis'.format(labelled_total))
    
    return points_clean, labelled_perc, labelled_total, include

# ----------------------------------------------------------------------------------------------------------------
def align_torso(points, vid_res_h, vid_res_w, framerate):
    #points: dataframe of x y coordinates ouput from DLC
    #vid_res_h: video heigth in pixels
    #vid_res_w: video width in pixels
    #framerate: video frame rate
    #get points for torso
    idx = pd.IndexSlice
    points_torso = points.loc[:,idx[:,['LShoulder','RShoulder','LHip','RHip'],:]]
    nrows = points.shape[0]
    data = pd.DataFrame(index=range(nrows), columns = points.columns)

    #define torso template, most representative frame of torso
    #set row with any nan values to nan
    points_torso.values[np.isnan(points_torso).any(axis=1),:] = np.nan
    nframes = np.size(points,0)
    med = points_torso.median(axis = 0)
    # find fame closest to median value of torso points
    index = np.nanargmin(np.sum(np.subtract(points_torso.values, np.tile(med, (nframes,1))), axis = 1), axis=0)
    template = points_torso.loc[index,:]

    mid_hip = (template.loc[:,'LHip'] + template.loc[:,'RHip'])/2
    mid_shoulder = (template.loc[:,'LShoulder'] + template.loc[:,'RShoulder'])/2
    template_y = mid_shoulder - mid_hip
    t2 = template_y.values/np.linalg.norm(template_y.values)
    M = rotation_matrix(template_y.values, [-1,0], [0,0])

    #get all bodypoints in torso
    new_torso = []
    bp = template.index.get_level_values(1).unique()
    for p in bp:
        rot_point = rotate_point(template.loc[:,template.index.get_level_values(1)==p].values, M)
        new_torso = np.concatenate((new_torso, rot_point), axis=None)
    rot_torso = pd.DataFrame(data = new_torso[np.newaxis,:], columns = points_torso.columns)
    mid_point = [np.mean(rot_torso.loc[:,rot_torso.columns.get_level_values(2)=='x'].values), 
                 np.mean(rot_torso.loc[:,rot_torso.columns.get_level_values(2)=='y'].values)]
    rot_torso = rot_torso - np.tile(np.array(mid_point), (1,4))

    # check if points within ellipse
    res_x = vid_res_w*0.07
    res_y = vid_res_h*0.10

    points_torso_centred = points_torso - points_torso.median(axis = 0)
    xpoints = points_torso_centred.loc[:,points_torso_centred.columns.get_level_values(2)=='x']
    ypoints = points_torso_centred.loc[:,points_torso_centred.columns.get_level_values(2)=='y']
    inside_torso = np.add(np.divide(np.power(xpoints,2), np.power(res_x,2)),
                    np.divide(np.power(ypoints,2),np.power(res_y,2)))
    index_torso = inside_torso > 1

    outliers_lrg =  pd.concat([index_torso,index_torso.rename(columns={'x':'y'})], axis = 1)
    outliers_lrg = outliers_lrg.reindex(['LShoulder', 'RShoulder', 'LHip', 'RHip'], axis = 1, level = 1)
    points_rol = points_torso.mask(cond = outliers_lrg)

    #filter points, using moving average filter
    window_size = int(framerate)
    points_rol = points_rol.rolling(window = window_size, min_periods=5, center=True, 
                                    win_type = 'blackman', axis=0).mean()

    #fill gaps
    torso_df = points_rol.interpolate(method = 'linear', limit = 5, axis =0)
    imp = IterativeImputer(max_iter=30, random_state=0)
    imp.fit(torso_df)
    pred = imp.transform(torso_df)
    torso_df.loc[:,:] = pred

    points_gf = points_torso.fillna(value = torso_df)

    #transform points
    x_torso = rot_torso.loc[:,rot_torso.columns.get_level_values(2)=='x'].values[0]
    y_torso = rot_torso.loc[:,rot_torso.columns.get_level_values(2)=='y'].values[0]
    template_rs = [x_torso,y_torso]
    torsofit_df, trans, res= map_torso(points_gf, template_rs)
    df_transform = transform_points(points, trans)
    points_adjcm = df_transform
    
    return points_adjcm 

def map_torso(points_raw, template):
# Outputs
# torsofit_df = 4 torso markers fitted position
# trans = transformation matrix  [R|d] R 2x2 rotation d 2x1 displacement, extra row added to make dims consistent
# res residual of fitting per frame

    nframes = np.size(points_raw,0)
    points_clean = points_raw
    nPoints = 4
    points_fit = np.empty((0,8))
    trans_mat = np.empty((3,3,0))
    res_mat = np.empty((2,nPoints,0))
    for f in range(0,nframes):
        frame = np.concatenate((np.atleast_2d(points_clean.loc[f:f,points_clean.columns.get_level_values(2)=='x'].values), 
                         np.atleast_2d(points_clean.loc[f:f,points_raw.columns.get_level_values(2)=='y'].values)), 
                        axis = 0)
    
        #check for missing markers
        missing = np.argwhere(np.isnan(frame))
        Rot, dis, Tran, res = ls_fit(template, frame)
        
        points_map = np.matmul(Tran,np.append(frame,np.atleast_2d([1,1,1,1]), axis=0))
        
        #manipilate matrix to original configuration
        point_remap = np.delete(points_map, 2, axis = 0)
        fit = np.atleast_2d(point_remap.flatten('F'))
        points_fit = np.append(points_fit, fit, axis = 0)
        trans_mat = np.append(trans_mat, np.atleast_3d(Tran), axis = 2)
        res_mat = np.append(res_mat, np.atleast_3d(res), axis = 2)
        
    col = points_raw.columns
    row = points_raw.index
    torso_df = pd.DataFrame(data = points_fit, index = row, columns = col)
    return (torso_df, trans_mat, res_mat)

def ls_fit(points, template):
    nPoints =  np.size(template, 1)
    
    #centreing, C = centre point, 0 = points - centre 
    targetC = np.transpose(np.atleast_2d(np.mean(template, axis=1)))
    target0 = np.subtract(template, np.tile(targetC, nPoints))
    
    pointsC = np.transpose(np.atleast_2d(np.mean(points, axis=1)))
    points0 = np.subtract(points, np.tile(pointsC, nPoints))
    
    #svd 
    C = np.atleast_2d(np.matmul(points0, np.transpose(target0)))
    u,s,v = np.linalg.svd(C, compute_uv=True)
    
    #calc rotation and displacement matrix
    RotMat = np.matmul(np.matmul(u, np.diagflat([1,np.linalg.det(np.matmul(u,np.transpose(v)))])), np.transpose(v))

    dispMat = np.transpose(np.atleast_2d(np.subtract(np.mean(points, axis=1),
                                                 np.matmul(RotMat,np.mean(template, axis=1)))))
 
    TransMat = np.concatenate((np.append(RotMat,np.atleast_2d([0,0]), axis=0), 
                 np.append(dispMat,np.atleast_2d([1]), axis=0)), axis = 1)

    #calculate residules 
    seg_gen = np.ones((1,nPoints))
    res = np.subtract(points, np.add(np.kron(dispMat,seg_gen), np.matmul(RotMat,template)))
    
    return(RotMat, dispMat, TransMat, res)

def transform_points(points, trans):
    #apply transformation to all points
    nframes = points.shape[0]
    col = points.columns
    row = points.index
    points_fit = np.empty((0,col.shape[0]))

    keypoints = ['Crown', 'LEye', 'REye', 'Chin', 'LShoulder', 'LElbow', 'LHand', 'LHip',
           'LKnee', 'LHeel', 'LToe', 'RShoulder', 'RElbow', 'RHand', 'RHip',
           'RKnee', 'RHeel', 'RToe']


    for f in range(0,nframes):
        fit_kp = np.empty((1,0))
        for k in enumerate(keypoints):
            frame = np.concatenate((np.atleast_2d(points.loc[f:f, 
                np.logical_and(points.columns.get_level_values(2)=='x', 
                points.columns.get_level_values(1)==k[1])].values), 
                np.atleast_2d(points.loc[f:f,np.logical_and(points.columns.get_level_values(2)=='y', 
                points.columns.get_level_values(1)==k[1])].values)), axis = 0)
            if not(np.any(np.isnan(frame))):
                points_map = np.matmul(trans[:,:,f],np.append(frame,np.atleast_2d([1]), axis=0))
                point_remap = np.delete(points_map, 2, axis = 0)
                fit = np.atleast_2d(point_remap.flatten('F'))
            else:
                fit = [[np.nan, np.nan]]
            fit_kp = np.append(fit_kp, fit, axis = 1)
        points_fit = np.append(points_fit, fit_kp, axis = 0)

    points_fit_df = pd.DataFrame(data = points_fit, index = row, columns = col)
    return (points_fit_df)

def rotation_matrix(v1, v2, translation):
    #Returns a 2D rotation matrix that rotates vector v1 to v2
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(det, dot)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    T = np.array(translation)
    M = np.identity(3)
    M[:2, :2] = R
    M[:2, 2] = T
    return M

def rotate_point(point, M):
    #Rotates a 2D point using a rotation matrix defined by v1 and v2, with an optional translation
    point = np.append(point, [1])
    point = np.dot(M, point)
    return point[:2]

# ----------------------------------------------------------------------------------------------------------------
def remove_outliers(points, vid_res_h, vid_res_w):
    #points: dataframe of x y coordinates ouput from DLC
    #vid_res_h: video heigth in pixels
    #vid_res_w: video width in pixels
    
    numpoints1 = np.sum(points.count())

    bodyparts = np.unique(points.columns.levels[1])
    nbp = len(bodyparts)
    
    #determine envelope of baby
    crown = points.loc[:,points.columns.get_level_values(1)=='Crown']
    LHip = points.loc[:,points.columns.get_level_values(1)=='LHip']
    RHip = points.loc[:,points.columns.get_level_values(1)=='RHip']
    midHip = (LHip.values + RHip.values)/2
    length_sq = np.power((crown.values - midHip),2)
    length = np.nanmedian(np.sqrt(length_sq[:,0] + length_sq[:,1]), axis =0)

    #define ellipse to remove outliers
    x_thres = length * 1.5
    y_thres = length * 1

    #centre points to midhip
    points_center_hip = points - np.tile(midHip, (1,nbp))

    xpoints_body = points_center_hip.loc[:,points_center_hip.columns.get_level_values(2)=='x']
    ypoints_body = points_center_hip.loc[:,points_center_hip.columns.get_level_values(2)=='y']

    inside_body = np.add(np.divide(np.power(xpoints_body,2), np.power(x_thres,2)),
                    np.divide(np.power(ypoints_body,2),np.power(y_thres,2)))
    index_body = inside_body > 1

    outliers_body =  pd.concat([index_body,index_body.rename(columns={'x':'y'})], axis = 1)
    outliers_body = outliers_body.reindex(bodyparts, axis = 1, level = 1)

    points_out_body = points.mask(cond = outliers_body)
    
    #remove outliers for individual bodyparts
    points_centred_bp = points_out_body - points_out_body.median(axis=0)
    xpoints_bp = points_centred_bp.loc[:,points_centred_bp.columns.get_level_values(2)=='x']
    ypoints_bp = points_centred_bp.loc[:,points_centred_bp.columns.get_level_values(2)=='y']

    #check if points within ellipse, deemed more y movement than x
    #maybe these threshold can be based on quartiles of the data o bodypart
    thresx2 = 0.2* vid_res_h
    thresy2 = 0.3 * vid_res_w
    inside_bp = np.add(np.divide(np.power(xpoints_bp,2), np.power(thresx2,2)),
                    np.divide(np.power(ypoints_bp,2),np.power(thresy2,2)))
    index_bp = inside_bp > 1

    outliers_bp =  pd.concat([index_bp,index_bp.rename(columns={'x':'y'})], axis = 1)
    outliers_bp = outliers_bp.reindex(bodyparts, axis = 1, level = 1)

    points_out_bp = points_out_body.mask(cond = outliers_bp)
    
    numpoints2 = np.sum(points_out_bp.count())
    num_removed = numpoints1 - numpoints2
    per_removed = 100*(numpoints1 - numpoints2)/numpoints1
    print('start {:} points, after outlier removal {:} points, {:.2}% removed'.format(numpoints1, 
                                                                                      numpoints2, per_removed))

    return points_out_bp, per_removed, num_removed

# ----------------------------------------------------------------------------------------------------------------
def gap_fill(points, framerate):
    #points: dataframe of x y coordinates ouput from DLC
    #framerate: video frame rate
    
    imp = IterativeImputer(max_iter=30, random_state=0)
    nframes = points.shape[0]
            
    #linear interpolation gap < 5
    points_fill = points.interpolate(method = 'linear', limit = 5, axis =0)

    #rolling average to do prediction on
    window_size = int(framerate)
    points_fill = points_fill.rolling(window = window_size, min_periods=5, center=True, win_type = 'blackman', axis=0).mean()

    #fill in mising values    
    imp.fit(points_fill)
    IterativeImputer(random_state=0)
    pred = imp.transform(points_fill)

    df_pred = pd.DataFrame(data = pred, index = points_fill.index, columns = points_fill.columns)
    points_fill2 = points.fillna(value = df_pred)
    
    return points_fill2
    
# ----------------------------------------------------------------------------------------------------------------
def size_normalise(points):
    #points: dataframe of x y coordinates ouput from DLC

    #determine average length of baby
    crown = points.loc[:,points.columns.get_level_values(1)=='Crown']
    LHip = points.loc[:,points.columns.get_level_values(1)=='LHip']
    RHip = points.loc[:,points.columns.get_level_values(1)=='RHip']
    midHip = (LHip.values + RHip.values)/2
    length_sq = np.power((crown.values - midHip),2)
    length = np.nanmedian(np.sqrt(length_sq[:,0] + length_sq[:,1]), axis =0)

    #use midhip as centre
    centrex = midHip[:,0][:,np.newaxis]
    centrey = midHip[:,1][:,np.newaxis]

    #adjust to make hips centre point
    nbp = len(points.columns.unique(level = 1))
    points_centre = points - np.tile(np.concatenate((centrex, centrey), axis =1), (1,nbp))

    #norm to baby length
    points_norm_length = points_centre / length
    
    return points_norm_length, length

# ----------------------------------------------------------------------------------------------------------------
def time_normalise(points):
    #points: dataframe of x y coordinates ouput from DLC

    norm_frames = 4500 #number of frames to normalise data to

    nframes, ncols = points.shape
    x = points.index
    tnorm_points = np.empty((norm_frames, 0))
    xnew = np.linspace(0,nframes-1, num = norm_frames, endpoint = True)

    for colName, colData in points.iteritems():
        y = colData.values
        fun = interp1d(x, y, kind = 'cubic')
        points_norm = fun(xnew)
        tnorm_points = np.concatenate((tnorm_points, points_norm[:,np.newaxis]), axis = 1)

    frames_new = range(0,norm_frames)
    df_timenorm = pd.DataFrame(data = tnorm_points, index = frames_new, columns = points.columns)    

    return df_timenorm

# ----------------------------------------------------------------------------------------------------------------


def plot_trajectories_scatter(points, output_dir, colors, options):
    #points: dataframe of x y coordinates
    #output_dir: folder to output  graph
    #options: options.height corresponds to long axis of video, from head to feet of baby 
    #options.width corresponds to short axis of video, from left to right of baby

    #figure properties
    wid=3
    ratio=options['height']/options['width']
    height=wid*ratio
    fig1=plt.figure(figsize=(wid, height), dpi=300) # need to adjust figure size based on video resolution 
    ax1=fig1.add_subplot(111)    
        
    step_dict = {1:'qc', 2:'align', 3:'outliers', 4:'gapfill', 5:'sizenorm', 6:'timenorm'}
    suffix = step_dict[options['step']]
    if options['step'] ==1:
        left = options['width']
        bottom = options['height']
        right = 0
        top = 0
        units = 'pixels'
    if options['step'] >= 2 and options['step'] <= 4:
        bottom = options['height']/2
        left = options['width']/2
        right = -left
        top = -bottom
        units = 'pixels'
    if options['step'] >= 5 and options['step'] <= 6:
        length = options['unit_length']
        bottom = options['height']/(2*length)
        left = options['width']/(2*length)
        right = -left
        top = -bottom
        units = 'infant unit length'

    bodypoints = list(points.columns.unique(level = 1))
    for bpindex, bp in enumerate(bodypoints):
        temp_y = np.ma.array(
            points.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze())
        temp_x = np.ma.array(
            points.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze())
        ax1.plot(temp_x, temp_y, ".", color=colors[bp], markersize=2)
    
    ax1.axis('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim(bottom = bottom, top = top)
    ax1.set_xlim(left = left, right = right)
    plt.tight_layout()
    

    fig1.savefig(output_dir + options['name']+'_trajscatter_' + suffix + '.png')

    
def plot_trajectories_xy(points, output_dir, colors, options):
    #points: dataframe of x y coordinates 
    #output_dir: folder to output  graph
    axis_size = 12

    #get x and y coordinates
    x = points.loc[:,points.columns.get_level_values(2)=='x']
    y = points.loc[:,points.columns.get_level_values(2)=='y']
    bodyparts = y.columns.get_level_values(1)

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,8), sharex = 'all')
    ax2.set_xlabel("Frame number", fontsize = axis_size)
    ax2.set_xlim(0,points.shape[0])

    bodypoints = list(points.columns.unique(level = 1))
    for bpindex, bp in enumerate(bodypoints):
        ax1.plot(y.values[:,bpindex], color=colors[bp])
        ax2.plot(x.values[:,bpindex], color=colors[bp])

    step_dict = {1:'qc', 2:'align', 3:'outliers', 4:'gapfill', 5:'sizenorm', 6:'timenorm'}
    suffix = step_dict[options['step']]
    if options['step'] >=1 and options['step'] <=4:
        xaxis = options['width']
        yaxis = options['height'] 
        units = 'pixels'
        ax1.set_ylim(0, xaxis)
        ax2.set_ylim(0, yaxis)
    if options['step'] >= 5 and options['step'] <= 6:
        length = options['unit_length']
        xaxis = options['width']/(2*length)
        yaxis = options['height']/(2*length)
        units = 'infant unit length'
        ax1.set_ylim(-xaxis, xaxis)
        ax2.set_ylim(-yaxis, yaxis)
        
                 
    ax1.set_ylabel("x-coordinate", fontsize = axis_size)
    ax2.set_ylabel("y-coordinate", fontsize = axis_size) 
    ax1.tick_params(axis='y', which='major', labelsize=axis_size)
    ax2.tick_params(axis='both', which='major', labelsize=axis_size)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    fig.savefig(output_dir + options['name']+'_trajplot_' + suffix + '.png')

# ----------------------------------------------------------------------------------------------------------------
#create video of trajectories
def create_trajvideo(points, colors, output_dir, voptions):
    frame_dir = trajectory_video(points, colors, output_dir, voptions)
    create_video(frame_dir, voptions['fps'])

#create video of trajectories
def trajectory_video(points, colors, output_dir, options):
    step_dict = {1:'qc', 2:'align', 3:'outliers', 4:'gapfill', 5:'sizenorm', 6:'timenorm'}
    suffix = step_dict[options['step']]
    
    vidname = options['name']
    for row in points.iterrows():
        f = row[0]
        outpath = output_dir+'trajvideo/'+suffix+'/'
        os.makedirs(outpath, exist_ok=True)
        name = 'frame_{:04d}'.format(f) 
#         print(name)
        # plot
        plot_scatter_video(row[1].to_frame().transpose(), outpath+name, colors, options)
    
    return outpath

def plot_scatter_video(points, output_base, colors, options):
    #points: dataframe of x y coordinates ouput from DLC
    #output_dir: folder to output scatter graph
    #options: options.length corresponds to long axis of video, from head to feet of baby 
    #options.width corresponds to short axis of video, from left to right of baby

    bodypoints = list(points.columns.unique(level = 1))

    # Pose X vs pose Y
    wid=2
    ratio = options['height']/options['width']
    height=wid*ratio
    fig1 = plt.figure(figsize=(wid, height), dpi=300) # need to adjust figure size based on video resolution 
    ax1 = fig1.add_subplot(111)
    
    step_dict = {1:'qc', 2:'align', 3:'outliers', 4:'gapfill', 5:'sizenorm', 6:'timenorm'}
    suffix = step_dict[options['step']]
        
    if options['step'] ==1:
        left = options['width']
        bottom = options['height']
        right = 0
        top = 0
        units = 'pixels'
    if options['step'] >= 2 and options['step'] <= 4:
        bottom = options['height']/2
        left = options['width']/2
        right = -left
        top = -bottom
        units = 'pixels'
    if options['step'] >= 5 and options['step'] <= 6:
        length = options['unit_length']
        bottom = options['height']/(2*length)
        left = options['width']/(2*length)
        right = -left
        top = -bottom
        units = 'infant unit length'

    for bpindex, bp in enumerate(bodypoints):
        temp_y = np.ma.array(
            points.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze())
        temp_x = np.ma.array(
            points.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze())
        ax1.plot(temp_x, temp_y, markerfacecolor='none', markeredgecolor=colors[bp], 
                 alpha=1, marker = 'o', markersize=5, markeredgewidth=2)

    #axis settings
    ax1.set_ylim(bottom = bottom, top = top)
    ax1.set_xlim(left = left, right = right)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout()
    
    fig1.savefig(output_base + '.jpg', transparent=False)
    plt.close()
    return 
    
def create_video(outpath, fps):
    frames = []
    imgs = glob.glob(outpath+'*.jpg')
    imgs.sort()
    for i in imgs:
        new_frame = cv2.imread(i)
        h, w, layers = new_frame.shape
        size = (w, h)
        frames.append(new_frame)

    out = cv2.VideoWriter(outpath+'trajvid.mp4',cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,size)

    for j in range(len(frames)):
        out.write(frames[j])
    out.release()
    
# ----------------------------------------------------------------------------------------------------------------    
# Video input
vid_dir = 'vid_dir'
suffix = 'suffix' #dlc suffix for video naming
file_names = [vid_dir + 'video.mp4']

output_dir = '/data/'

# Alternative if you wish to run on group of videos within the same folder
# file_names = glob.glob(vid_dir + '*.mp4')
# file_names = [x for x in file_names if not 'labeled' in x]

alpha = 1
colors = {'Crown':(0.27,0.01,0.34,alpha),
        'LEye':(0.28,0.16,0.47,alpha),
        'REye':(0.28,0.16,0.47,alpha),
        'Chin':(0.23,0.31,0.54,alpha),
        'LShoulder':(0.18,0.42,0.55,alpha),
        'RShoulder':(0.18,0.42,0.55,alpha),
        'LElbow':(0.17,0.45,0.56,alpha),
        'RElbow':(0.17,0.45,0.56,alpha),
        'LHand':(0.11,0.59,0.54,alpha),
        'RHand':(0.11,0.59,0.54,alpha),
        'LHip':(0.27,0.75,0.43,alpha),
        'RHip':(0.27,0.75,0.43,alpha),
        'LKnee':(0.56,0.84,0.26,alpha),
        'RKnee':(0.56,0.84,0.26,alpha),
        'LHeel':(0.82,0.89,0.11,alpha),
        'RHeel':(0.82,0.89,0.11,alpha),
        'LToe':(1,0.91,0.14,alpha),
        'RToe':(1,0.91,0.14,alpha)}

for video in file_names:
    process_points = []
    #Load dlc x y output from csv file
    path, file = os.path.split(video)
    base_name = file.split('.')[0]
    results_dir = output_dir+base_name+'/'
    print('processing file {}'.format(file))
    
    #load csv file
    point_file = path + '/' + base_name + suffix + '.csv'
    df_points = pd.read_csv(point_file, index_col = 0, header = [0,1,2])
    #load hd5 file - alternative
#     point_file = path + '/' + base_name + suffix + '.h5'
#     df_points = pd.read_hdf(point_file, index_col = 0, header = [0,1,2])
    
    #load video file
    vid = cv2.VideoCapture(video)
    height = vid.get(cv2.CAP_PROP_FRAME_WIDTH )
    width = vid.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vid.get(cv2.CAP_PROP_FPS)
    framerate = vid.get(cv2.CAP_PROP_FPS)
    print('video resoultion: {} x {}, frame rate: {}'.format(height, width, fps))

    #---------------------------------------------------------------------------------------------
    print('Step 1 - quality control')
    likelihood = 0.2 # deep lab cut likelihood value minimum to keep
    qc = 70 # minimum % of body points labelled to include
    points_qc, labelled_perc, labelled_total, include = quality_control(df_points, likelihood, qc)
    #points_qc.to_csv(results_dir+'points_qc.csv')
    num_steps = 1
    # step 2
    if include:
        print('Step 2 - camera movement adjustment')
        #align torso to midline
        points_aligned = align_torso(points_qc, height, width, fps)
        #points_aligned.to_csv(results_dir+'points_aligned.csv')
        
        print('Step 3 - remove outliers')
        points_outliersrm, per_removed, num_removed = remove_outliers(points_aligned, height, width)
        #points_outliersrm.to_csv(results_dir+'points_outliersrm.csv')
        
        print('Step 4 - Gap filling')
        points_filt = gap_fill(points_outliersrm, fps)
        #points_filt.to_csv(results_dir+'points_gapfill.csv')
        
        print('Step 5 - Size normalisation')
        points_norm1, infant_length = size_normalise(points_filt)
        #points_norm1.to_csv(results_dir+'points_norm1.csv')
        sfh = height/infant_length
        sfw = width/infant_length
        print('scale factor based on infant length {:.2}: new deminsions w{:.2} x h{:.2}'.format(infant_length, sfw, sfh))

        print('Step 6 - Time normalisation')
        points_norm2 = time_normalise(points_norm1)
        #points_norm2.to_csv(results_dir+'points_norm2.csv')
        
        print('finished processing')
        
        #create graph of results
#         proc_points = {1: points_qc, 2:points_aligned, 3:points_outliersrm, 4:points_filt, 5:points_norm1, 
#                6:points_norm2}
#         step = 2
#         options = {'name': base_name, 'height': height, 'width': width, 'unit_length': infant_length, 
#                'step': step, 'save': figsave}
#         fig1 = plot_trajectories_scatter(proc_points[step], results_dir, colors, options)
#         fig2 = plot_trajectories_xy(proc_points[step], results_dir, colors, options)
        
        #create video of results
#         voptions = {'name': base_name, 'height': height, 'width': width, 'fps':fps, 'unit_length': infant_length, 
#                        'step': s}
#         vid_path = create_trajvideo(proc_points[step], colors, result_dir, voptions)
    

