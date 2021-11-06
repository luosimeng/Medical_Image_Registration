import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=1, help='Size of minibatch')  #default=4 
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tflearn

import network
import data_util.liver
# import data_util.brain


def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get(
        'segmentation_class_value', None), fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args.dataset, batch_size=args.batch, paired=args.paired, **
                 eval('dict({})'.format(args.data_args)))

    sess = tf.Session()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES))
        checkpoint = args.checkpoint
        saver.restore(sess, checkpoint)
        tflearn.is_training(False, session=sess)

        val_subsets = [data_util.liver.Split.VALID]  #=[2]
        if args.val_subset is not None:
            val_subsets = args.val_subset.split(',')

        tflearn.is_training(False, session=sess)
        keys = ['pt_mask', 'landmark_dist', 'jaccs', 'dices', 'jacobian_det']
        if not os.path.exists('evaluate'):
            os.mkdir('evaluate')
        path_prefix = os.path.join('evaluate', short_name(checkpoint))
        if args.rep > 1:
            path_prefix = path_prefix + '-rep' + str(args.rep)
        if args.name is not None:
            path_prefix = path_prefix + '-' + args.name
        step_i=0
        for val_subset in val_subsets:  
            step_i += 1
            if args.val_subset is not None:
                output_fname = path_prefix + '-' + str(val_subset) + '.txt'
            else:
                output_fname = path_prefix + '.txt'

            with open(output_fname, 'w') as fo:
                print("Validation subset {}".format(val_subset))
                gen = ds.generator(val_subset, loop=False)
                results = framework.validate(step_i,sess, gen, summary=False, show_tqdm=True)
                for i in range(len(results['dicem'])):
                    print(results['id1'][i], results['id2'][i], np.mean(results['dicem'][i]), file=fo)
                print('Summary', file=fo)
                print("Dice score: {} ({})".format(np.mean(results['dicem']), np.std(np.mean(results['dicem'], axis=-1))), file=fo)
                print('jaccardm:',np.mean(results['jaccardm']), file=fo)
                print('hdm:',np.mean(results['hdm']), file=fo)
                print('assdm:',np.mean(results['assdm']), file=fo)
                print('precisionm:',np.mean(results['precisionm']), file=fo)
                print('sensitivitym:',np.mean(results['sensitivitym']), file=fo)
                print('specificitym:',np.mean(results['specificitym']), file=fo)
        

def short_name(checkpoint):
    cpath, steps = os.path.split(checkpoint)
    _, exp = os.path.split(cpath)
    return exp + '-' + steps


def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


if __name__ == '__main__':
    main()
