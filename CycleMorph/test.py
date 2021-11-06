import os
import numpy as np
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch
import scipy.io as sio
import scipy.ndimage
from models.networks import Dense3DSpatialTransformer
from medipy.metrics import dice
import torch.nn.functional as F
import nibabel as nib
from random import choice
import medpy.metric.binary as medpy


dice_means = []
dice_sep = []
jaccard_means = []
hd_means = []
assd_means = []
precision_means = []
sensitivity_means = []
specificity_means = []

def metrics_7(array1, array2, labels=None, include_zero=False):
    """
    Computes 7 metrics overlap between two arrays for a given set of integer labels.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 
    dicem = np.zeros(len(labels))
    jaccardm = np.zeros(len(labels))
    hdm = np.zeros(len(labels))
    assdm = np.zeros(len(labels))
    precisionm = np.zeros(len(labels))
    sensitivitym = np.zeros(len(labels))
    specificitym = np.zeros(len(labels))

    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
        jaccardm[idx] = medpy.jc((array1 == label).astype(float),(array2 == label).astype(float))
        try:
            hdm[idx] = medpy.hd((array1 == label).astype(float),(array2 == label).astype(float))
            assdm[idx] = medpy.assd((array1 == label).astype(float),(array2 == label).astype(float))
        except:
            hdm[idx] = 311.0
            assdm[idx] = 311.0
        precisionm[idx] = medpy.precision((array1 == label).astype(float),(array2 == label).astype(float))
        sensitivitym[idx] = medpy.sensitivity((array1 == label).astype(float),(array2 == label).astype(float))
        specificitym[idx] = medpy.specificity((array1 == label).astype(float),(array2 == label).astype(float))

    return np.mean(dicem),np.mean(jaccardm),np.mean(hdm),np.mean(assdm),np.mean(precisionm),np.mean(sensitivitym),np.mean(specificitym)


def my_dice(array1, array2, labels=None, include_zero=False):
    """
    Computes 7 metrics overlap between two arrays for a given set of integer labels.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 
    dicem = np.zeros(len(labels))

    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
        
    return np.mean(dicem)

def _toTorchFloatTensor(img):
    img = torch.from_numpy(img.copy())
    return img

def _transform(dDepth, dHeight, dWidth):
    batchSize = dDepth.shape[0]
    dpt = dDepth.shape[1]
    hgt = dDepth.shape[2]
    wdt = dDepth.shape[3]

    D_mesh = torch.linspace(0.0, dpt - 1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt)
    h_t = torch.matmul(torch.linspace(0.0, hgt - 1.0, hgt).unsqueeze_(1), torch.ones((1, wdt)))
    H_mesh = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
    w_t = torch.matmul(torch.ones((hgt, 1)), torch.linspace(0.0, wdt - 1.0, wdt).unsqueeze_(1).transpose(1, 0))
    W_mesh = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)

    D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    D_upmesh = dDepth.float() + D_mesh
    H_upmesh = dHeight.float() + H_mesh
    W_upmesh = dWidth.float() + W_mesh
    return torch.stack([D_upmesh, H_upmesh, W_upmesh], dim=1)

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.nThreads = 1
    opt.batchSize = 1
    model_regist = create_model(opt)
    stn = Dense3DSpatialTransformer()


    datafiles = []  #路径的list
    dataFiles = sorted(os.listdir(opt.dataroot))

    with open("/media/gdp/work/lsm/CycleMorph/Pair_list.txt", "r") as ftxt:
        for line in ftxt.readlines():
            line = line.strip('\n')
            index_m,index_f=line.split(' ')

            print('moving volume = %s' % (index_m))
            print('fixed volume = %s' % (index_f))

            data_vol=nib.load('/media/gdp/work/gxr/ViT-V-Net/alignednorm_data/aligned_norm'+index_m+'.nii.gz').get_data()
            # data_seg_path='/media/gdp/work/lsm/CycleMorph/seg35_data/aligned_seg35_'+dataFile[24:28]+'.nii.gz'
            data_seg=nib.load('/media/gdp/work/gxr/ViT-V-Net/align_seg35_data/aligned_seg35_'+index_m+'.nii.gz').get_data()

            label_vol=nib.load('/media/gdp/work/gxr/ViT-V-Net/alignednorm_data/aligned_norm'+index_f+'.nii.gz').get_data()
            # label_seg_path='/media/gdp/work/lsm/CycleMorph/seg35_data/aligned_seg35_'+dataFile_f[24:28]+'.nii.gz'
            label_seg=nib.load('/media/gdp/work/gxr/ViT-V-Net/align_seg35_data/aligned_seg35_'+index_f+'.nii.gz').get_data()

            test_dataS = data_seg.transpose(2, 1, 0).astype(float)  # D W H
            nd = test_dataS.shape[0]
            nw = test_dataS.shape[1]
            nh = test_dataS.shape[2]
            test_dataS = test_dataS.reshape(1, 1, nd, nw, nh)
            batch_s = _toTorchFloatTensor(test_dataS)

            dataA = data_vol
            dataB = label_vol
            test_dataA = dataA.transpose(2, 1, 0).astype(float)
            test_dataB = dataB.transpose(2, 1, 0).astype(float)
            test_dataA = test_dataA.reshape(1, 1, nd, nw, nh)
            test_dataB = test_dataB.reshape(1, 1, nd, nw, nh)
            batch_x = _toTorchFloatTensor(test_dataA)
            batch_y = _toTorchFloatTensor(test_dataB)
            ###################################################

            test_data = {'A': batch_x, 'B': batch_y, 'path': '_'}
            model_regist.set_input(test_data)
            model_regist.test()
            visuals = model_regist.get_test_data()
            regist_flow = visuals['flow_A'].cpu().float().numpy()[0].transpose(3, 2, 1, 0)

            global_flow = regist_flow.transpose(3, 2, 1, 0)
            global_flow = _toTorchFloatTensor(global_flow).unsqueeze(0)
            regist_data = stn(batch_x.cuda().float(), global_flow.cuda().float())
            regist_data = regist_data.cpu().float().numpy()[0, 0].transpose(2, 1, 0)

            sflow = _transform(global_flow[:, 0], global_flow[:, 1], global_flow[:, 2])
            nb, nc, nd, nw, nh = sflow.shape
            segflow = torch.FloatTensor(sflow.shape).zero_()
            segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0
            segflow[:, 1] = (sflow[:, 1] / (nw - 1) - 0.5) * 2.0
            segflow[:, 0] = (sflow[:, 2] / (nh - 1) - 0.5) * 2.0
            regist_seg = F.grid_sample(batch_s.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)), mode='nearest')
            regist_seg = regist_seg.cpu().numpy()[0, 0].transpose(2, 1, 0)


            # dicem,jaccardm,hdm,assdm,precisionm,sensitivitym,specificitym= metrics_7(regist_seg, label_seg)
            dicem = my_dice(regist_seg, label_seg)
            dice_means.append(dicem)
            # jaccard_means.append(jaccardm)
            # hd_means.append(hdm)
            # assd_means.append(assdm)
            # precision_means.append(precisionm)
            # sensitivity_means.append(sensitivitym)
            # specificity_means.append(specificitym)
            # print(dicem)

            # #save images
            # new_image=nib.Nifti1Image(data_vol.squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/img_moving.nii.gz')
            # new_image=nib.Nifti1Image(data_seg.astype(float).squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/seg_moving.nii.gz')
            # new_image=nib.Nifti1Image(label_vol.squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/img_fixed.nii.gz')
            # new_image=nib.Nifti1Image(label_seg.astype(float).squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/seg_fixed.nii.gz')
            # new_image=nib.Nifti1Image(regist_data.squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/img_warped.nii.gz')
            # new_image=nib.Nifti1Image(regist_seg.astype(float).squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/seg_warped.nii.gz')
            # new_image=nib.Nifti1Image(regist_flow.squeeze(),np.eye(4))
            # nib.save(new_image,'/media/gdp/work/lsm/CycleMorph/try_test_image/flow.nii.gz')

    print('************************')
    print('dice:',np.mean(dice_means),'±',np.std(dice_means))
    print('dice-max:',np.max(dice_means))
    print('dice-min:',np.min(dice_means))
    # print('jaccard:',np.mean(jaccard_means),'±',np.std(dice_means))
    # print('hd:',np.mean(hd_means),'±',np.std(dice_means))
    # print('assd:',np.mean(assd_means),'±',np.std(dice_means))
    # print('precision:',np.mean(precision_means),'±',np.std(dice_means))
    # print('sensitivity:',np.mean(sensitivity_means),'±',np.std(dice_means))
    # print('specificity:',np.mean(specificity_means),'±',np.std(dice_means))