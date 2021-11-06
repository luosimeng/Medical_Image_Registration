# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import tflearn
from tqdm import tqdm
import nibabel as nib
import SimpleITK as sitk
# from medpy.metric import binary
import medpy.metric.binary as medpy
import scipy.ndimage

from . import transform
from .utils import MultiGPUs
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .recursive_cascaded_networks import RecursiveCascadedNetworks
import voxelmorph as vxm

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def mask_metrics(seg1,seg2):
    dice_all=[]
    jacc_all=[]
    for i in range(1,36):
    # for i in [ 1, 2, 3, 5, 6, 13, 20, 21, 22, 24 , 25]:
        dice_score=medpy.dc(seg1==i,seg2==i)
        jacc_score=medpy.jc(seg1==i,seg2==i)
        dice_all.append(dice_score)
        jacc_all.append(jacc_score)
    return np.array([np.mean(dice_all)]),np.array([np.mean(jacc_all)])

def my_dice(seg1,seg2):
    dice_all=[]
    for i in range(1,36):
        dice_score=medpy.dc(seg1==i,seg2==i)
        dice_all.append(dice_score)
    return np.array([np.mean(dice_all)]),np.array(dice_all)

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

    return np.array([np.mean(dicem)]),np.array([np.mean(jaccardm)]),np.array([np.mean(hdm)]),np.array([np.mean(assdm)]),np.array([np.mean(precisionm)]),np.array([np.mean(sensitivitym)]),np.array([np.mean(specificitym)])


def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


def masked_mean(arr, mask):
    return tf.reduce_sum(arr * mask) / (tf.reduce_sum(mask) + 1e-9)


class FrameworkUnsupervised:
    net_args = {'class': RecursiveCascadedNetworks}  
    framework_name = 'gaffdfrm'

    def __init__(self, devices, image_size, segmentation_class_value, validation=False, fast_reconstruction=False):
        network_class = self.net_args.get('class', RecursiveCascadedNetworks)
        self.summaryType = self.net_args.pop('summary', 'basic')

        self.reconstruction = Fast3DTransformer() if fast_reconstruction else Dense3DSpatialTransformer()
        self.seg_reconstruction = vxm.networks.Transform([160,192,224], interp_method='nearest')


        # input place holder
        img1 = tf.placeholder(dtype=tf.float32, shape=[
                              None,160,192,224, 1], name='voxel1')
        img2 = tf.placeholder(dtype=tf.float32, shape=[
                              None,160,192,224, 1], name='voxel2')
        seg1 = tf.placeholder(dtype=tf.float32, shape=[
                              None,160,192,224, 1], name='seg1')
        seg2 = tf.placeholder(dtype=tf.float32, shape=[
                              None,160,192,224, 1], name='seg2')
        point1 = tf.placeholder(dtype=tf.float32, shape=[
                                None, 6, 3], name='point1')
        point2 = tf.placeholder(dtype=tf.float32, shape=[
                                None, 6, 3], name='point2')

        bs = tf.shape(img1)[0]   #batchsize
        augImg1, preprocessedImg2 = img1, img2    # img1 -> fixed

        aug = self.net_args.pop('c', None)
        if aug is None:
            imgs = img1.shape.as_list()[1:4]  #[128,156,180]
            control_fields = transform.sample_power(
                -0.4, 0.4, 3, tf.stack([bs, 5, 5, 5, 3])) * (np.array(imgs) // 40)
            augFlow = transform.free_form_fields(imgs, control_fields) #5d tensor with 3 channels `(batch_size, x, y, z, 3)

            def augmentation(x):
                return tf.cond(tflearn.get_training_mode(), lambda: self.reconstruction([x, augFlow]),
                               lambda: x)

            def augmenetation_pts(incoming):
                def aug(incoming):
                    aug_pt = tf.cast(transform.warp_points(
                        augFlow, incoming), tf.float32)
                    pt_mask = tf.cast(tf.reduce_all(
                        incoming >= 0, axis=-1, keep_dims=True), tf.float32)
                    return aug_pt * pt_mask - (1 - pt_mask)
                return tf.cond(tflearn.get_training_mode(), lambda: aug(incoming), lambda: incoming)
            
            augImg2 = augmentation(preprocessedImg2)
            augSeg2 = augmentation(seg2)
            augPt2 = augmenetation_pts(point2)
        elif aug == 'identity':
            augFlow = tf.zeros(
                tf.stack([tf.shape(img1)[0],160,192,224, 3]), dtype=tf.float32)
            augImg2 = preprocessedImg2
            augSeg2 = seg2
            augPt2 = point2
        else:
            raise NotImplementedError('Augmentation {}'.format(aug))

        learningRate = tf.placeholder(tf.float32, [], 'learningRate')
        if not validation:
            adamOptimizer = tf.train.AdamOptimizer(learningRate)

        self.segmentation_class_value = segmentation_class_value  
        self.network = network_class(
            self.framework_name, framework=self, fast_reconstruction=fast_reconstruction, **self.net_args)
        net_pls = [augImg1, augImg2, seg1, augSeg2, point1, augPt2]  
        if devices == 0:
            with tf.device("/cpu:0"):
                self.predictions = self.network(*net_pls)
                if not validation:
                    self.adamOpt = adamOptimizer.minimize(
                        self.predictions["loss"])
        else:
            gpus = MultiGPUs(devices)
            if validation:
                self.predictions = gpus(self.network, net_pls)
            else:
                self.predictions, self.adamOpt = gpus(
                    self.network, net_pls, opt=adamOptimizer)   #核心！！！
        self.build_summary(self.predictions) 
        self.summary_images(self.predictions)

    @property
    def data_args(self):
        return self.network.data_args

    def summary_images(self,predictions):
        tf.summary.image('moving',predictions['image_moving'][:,:,:,64,:])
        for k in predictions:
            if k.find('warped_moving') != -1:
                tf.summary.image(k, predictions[k][:,:,:,64,:])
        tf.summary.image('fixed',predictions['image_fixed'][:,:,:,64,:])
        self.summaryImages = tf.summary.merge_all()

    def build_summary(self, predictions):                 #写入tensorboard的信息
        self.loss = tf.reduce_mean(predictions['loss'])
        for k in predictions:
            if k.find('loss') != -1:
                tf.summary.scalar(k, tf.reduce_mean(predictions[k]))
        self.summaryOp = tf.summary.merge_all()

        if self.summaryType == 'full':
            tf.summary.scalar('dice_score', tf.reduce_mean(
                self.predictions['dice_score']))
            tf.summary.scalar('landmark_dist', masked_mean(
                self.predictions['landmark_dist'], self.predictions['pt_mask']))
            preds = tf.reduce_sum(
                tf.cast(self.predictions['jacc_score'] > 0, tf.float32))
            tf.summary.scalar('jacc_score', tf.reduce_sum(
                self.predictions['jacc_score']) / (preds + 1e-8))
            self.summaryExtra = tf.summary.merge_all()
        else:
            self.summaryExtra = self.summaryOp

    def get_predictions(self, *keys):
        return dict([(k, self.predictions[k]) for k in keys])

    # def get_images(self,keys):
    #     return dict([(k, self.predictions[k]) for k in keys])

    def validate_clean(self, sess, generator, keys=None):
        for fd in generator:
            _ = fd.pop('id1')
            _ = fd.pop('id2')
            _ = sess.run(self.get_predictions(*keys),
                         feed_dict=set_tf_keys(fd))

    def validate(self, steps, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        if keys is None:
            # keys = ['landmark_dist', 'pt_mask','dice_score','jacc_score','jacobian_det']#'dice_score','jacc_score'
            keys=['dicem','jaccardm','hdm','assdm','precisionm','sensitivitym','specificitym']
            # if self.segmentation_class_value is not None:
            #     for k in self.segmentation_class_value:
            #         keys.append('jacc_{}'.format(k))
        full_results = dict([(k, list()) for k in keys])
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['img1'] = []
                full_results['img2'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        record_i=0
        real_i=0

        dice_means = []
        jaccard_means = []
        hd_means = []
        assd_means = []
        precision_means = []
        sensitivity_means = []
        specificity_means = []

        for fd in generator:
            record_i+=1
            if(record_i>166):
                break
 
            id1 = fd.pop('id1')
            id2 = fd.pop('id2')
            
            real_i += 1
      
            #这里要改的，train和val根据需要不一样
            # keys = ['pt_mask', 'landmark_dist', 'jaccs', 'dices', 'jacobian_det','image_moving','image_fixed','seg_moving','seg_fixed','warped_moving','warped_moving_0','warped_moving_1','warped_moving_2','warped_moving_3','warped_seg_binary','real_flow'] 
            keys = ['seg_fixed','seg_moving','real_flow','warped_moving'] 
            results = sess.run(self.get_predictions(*keys), feed_dict=set_tf_keys(fd))
           
            seg_moving = results['seg_moving']
            seg_fixed = results['seg_fixed'].squeeze()
            # image_moving = results['image_moving'].squeeze()
            # image_fixed = results['image_fixed'].squeeze()
            # warped_seg_binary = results['warped_seg_binary'].squeeze()
            warped_moving = results['warped_moving'].squeeze()
            # warped_moving_0 = results['warped_moving_0'].squeeze()
            # warped_moving_1 = results['warped_moving_1'].squeeze()
            # warped_moving_2 = results['warped_moving_2'].squeeze()
            # warped_moving_3 = results['warped_moving_3'].squeeze()
            real_flow = results['real_flow']

            #try using transform function from vxm
            warped_seg_binary=self.seg_reconstruction.predict([seg_moving, real_flow]).squeeze()


            # # if (record_i%):
            # img_save=nib.Nifti1Image(seg_moving.squeeze(),np.eye(4))
            # nib.save(img_save,'/media/gdp/work/lsm/full_cascaded_oasis/try_test_image/'+str(record_i)+'seg_moving.nii.gz')
            # img_save=nib.Nifti1Image(seg_fixed,np.eye(4))
            # nib.save(img_save,'/media/gdp/work/lsm/full_cascaded_oasis/try_test_image/'+str(record_i)+'seg_fixed.nii.gz')
            # # img_save=nib.Nifti1Image(image_moving,np.eye(4))
            # # nib.save(img_save,'/data1/lsm/cascaded_oasis/test_output_images/'+str(record_i)+'moving.nii.gz')
            # img_save=nib.Nifti1Image(warped_seg_binary,np.eye(4))
            # nib.save(img_save,'/media/gdp/work/lsm/full_cascaded_oasis/try_test_image/'+str(record_i)+'seg_warped.nii.gz')
            # img_save=nib.Nifti1Image(warped_moving,np.eye(4))
            # nib.save(img_save,'/media/gdp/work/lsm/full_cascaded_oasis/try_test_image/'+str(record_i)+'warped.nii.gz')
            # # # img_save=nib.Nifti1Image(image_fixed,np.eye(4))
            # # # nib.save(img_save,'/data1/lsm/cascaded_oasis/test_output_images/'+str(record_i)+'fixed.nii.gz')
            # img_save=nib.Nifti1Image(real_flow.squeeze(),np.eye(4))
            # nib.save(img_save,'/media/gdp/work/lsm/full_cascaded_oasis/try_test_image/'+str(record_i)+'flow.nii.gz')
            
            # results['dice_score'],sep_dice_i=my_dice(seg_fixed,warped_seg_binary)
            dicem,jaccardm,hdm,assdm,precisionm,sensitivitym,specificitym= metrics_7(warped_seg_binary, seg_fixed)
            results['dicem']=dicem
            results['jaccardm']=jaccardm
            results['hdm']=hdm
            results['assdm']=assdm
            results['precisionm']=precisionm
            results['sensitivitym']=sensitivitym
            results['specificitym']=specificitym

            _ = results.pop('seg_fixed')
            # _ = results.pop('image_fixed')
            # _ = results.pop('warped_seg_binary')
            _ = results.pop('warped_moving')
            # _ = results.pop('warped_moving_0')
            # _ = results.pop('warped_moving_1')
            # _ = results.pop('warped_moving_2')
            # _ = results.pop('warped_moving_3')
            _ = results.pop('seg_moving')
            # _ = results.pop('image_moving')
            _ = results.pop('real_flow')

            # keys = ['dice_score', 'landmark_dist', 'pt_mask', 'jacc_score','jacobian_det']
            keys=['dicem','jaccardm','hdm','assdm','precisionm','sensitivitym','specificitym']
            
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['img1'] = fd['voxel1']
                    results['img2'] = fd['voxel2']
            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                full_results[k].append(v[mask])

        # np.savetxt('/media/gdp/work/lsm/full_cascaded_oasis/sep_dice.csv', np.array(sep_dice), delimiter=',')
        # np.savetxt('/media/gdp/work/lsm/full_cascaded_oasis/avg_dice.csv', np.array(avg_dice), delimiter=',')

        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if summary:
                full_results[k] = full_results[k].mean()
        
        return full_results
