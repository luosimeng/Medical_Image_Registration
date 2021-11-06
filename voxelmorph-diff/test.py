#!/usr/bin/env python

"""
Example script for testing quality of trained VxmDense models. This script iterates over a list of
images pairs, registers them, propagates segmentations via the deformation, and computes the dice
overlap. Example usage is:

    test.py  \
        --model model.h5  \
        --pairs pairs.txt  \
        --img-suffix /img.nii.gz  \
        --seg-suffix /seg.nii.gz

Where pairs.txt is a text file with line-by-line space-seperated registration pairs.
This script will most likely need to be customized to fit your data.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import time
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import nibabel as  nib
import medpy.metric.binary as medpy


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--model', required=True, help='VxmDense model file')
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# sanity check on input pairs
# if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
#     print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
#     exit(1)
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load seg labels if provided
labels = np.load(args.labels) if args.labels else None

# check if multi-channel data
add_feat_axis = not args.multichannel

# keep track of all dice scores
reg_times = []
dice_means = []
jaccard_means = []
hd_means = []
assd_means = []
precision_means = []
sensitivity_means = []
specificity_means = []

with tf.device(device):

    # load model and build nearest-neighbor transfer model
    registration_model = vxm.networks.VxmDense.load(args.model).get_registration_model()
    inshape = registration_model.inputs[0].shape[1:-1]
    transform_model = vxm.networks.Transform(inshape, interp_method='nearest')
    vol_transform_model = vxm.networks.Transform(inshape, interp_method='linear')

    for i in range(len(img_pairs)):

        # load moving image and seg
        moving_vol = vxm.py.utils.load_volfile(
            '/media/gdp/work/lsm/voxelmorph-liver/test_data_lits/img'+img_pairs[i][0]+'.nii.gz', np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_vol=moving_vol/255
        moving_seg = vxm.py.utils.load_volfile(
            '/media/gdp/work/lsm/voxelmorph-liver/test_data_lits/seg'+seg_pairs[i][0]+'.nii.gz', np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg=(moving_seg==255)
        
        # load fixed image and seg
        fixed_vol = vxm.py.utils.load_volfile(
            '/media/gdp/work/lsm/voxelmorph-liver/test_data_lits/img'+img_pairs[i][1]+'.nii.gz', np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_vol=fixed_vol/255
        fixed_seg = vxm.py.utils.load_volfile(
            '/media/gdp/work/lsm/voxelmorph-liver/test_data_lits/seg'+seg_pairs[i][1]+'.nii.gz', np_var='seg')
        fixed_seg=(fixed_seg==255)

        # predict warp and time
        start = time.time()
        warp = registration_model.predict([moving_vol, fixed_vol])
        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)

        # apply transform
        warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
        warped_img= vol_transform_model.predict([moving_vol, warp]).squeeze()

        # #save_images
        # new_img=nib.Nifti1Image(moving_seg.astype(float).squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/seg_moving.nii.gz')
        # new_img=nib.Nifti1Image(moving_vol.squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/img_moving.nii.gz')
        # new_img=nib.Nifti1Image(fixed_vol.squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/img_fixed.nii.gz')
        # new_img=nib.Nifti1Image(fixed_seg.astype(float).squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/seg_fixed.nii.gz')
        # new_img=nib.Nifti1Image(warped_img.squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/warped.nii.gz')
        # new_img=nib.Nifti1Image(warp.squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/flow.nii.gz')
        # new_img=nib.Nifti1Image(warped_seg.astype(float).squeeze(),np.eye(4))
        # nib.save(new_img,'/media/gdp/work/lsm/voxelmorph-liver/try_test_image/seg_warped.nii.gz')


        # compute volume overlap (dice)
        # overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=labels)
        # dice_means.append(np.mean(overlap))
        # print('Lits-Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (i + 1, reg_time,
        #                                                             np.mean(overlap),
        #                                                             np.std(overlap)))
        dicem,jaccardm,hdm,assdm,precisionm,sensitivitym,specificitym= vxm.py.utils.metrics_7(warped_seg, fixed_seg, labels=labels)
        dice_means.append(dicem)
        jaccard_means.append(jaccardm)
        hd_means.append(hdm)
        assd_means.append(assdm)
        precision_means.append(precisionm)
        sensitivity_means.append(sensitivitym)
        specificity_means.append(specificitym)


# print()
# print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
#                                                                     np.std(reg_times)))
# print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))

# np.savetxt('/media/gdp/work/lsm/voxelmorph-liver/lits_test_dice.csv', np.array(dice_means), delimiter=',')

print('dice:',np.mean(dice_means),'±',np.std(dice_means))
print('jaccard:',np.mean(jaccard_means),'±',np.std(dice_means))
print('hd:',np.mean(hd_means),'±',np.std(dice_means))
print('assd:',np.mean(assd_means),'±',np.std(dice_means))
print('precision:',np.mean(precision_means),'±',np.std(dice_means))
print('sensitivity:',np.mean(sensitivity_means),'±',np.std(dice_means))
print('specificity:',np.mean(specificity_means),'±',np.std(dice_means))