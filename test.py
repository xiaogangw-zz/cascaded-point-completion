import argparse
import importlib
import numpy as np
import tensorflow as tf
import models
import time
import h5py
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append('./utils')
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
import tf_util

objects = ['plane', 'cabinet', 'car', 'chair', 'lamp', 'couch', 'table', 'watercraft', 'speaker', 'firearm','cellphone', 'bench', 'monitor']
snc_synth_id_to_category = {
    '02691156': 'plane',  # 3795
    '02828884': 'bench',
    '02933112': 'cabinet',  # 1322
    '02958343': 'car',  # 5766
    '03001627': 'chair',  # 5750
    '03211117': 'monitor',
    '03636649': 'lamp',  # 2068
    '03691459': 'speaker',
    '04090263': 'firearm',
    '04256520': 'couch',  # 2923
    '04379243': 'table',  # 5750
    '04401088': 'cellphone',
    '04530566': 'watercraft'  # 1689
}

def test(args):
    inputs = tf.placeholder(tf.float32, (1, 2048, 3))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    reconstruction = tf.placeholder(tf.float32, (1, args.step_ratio*1024, 3))
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    mean_feature = tf.placeholder(tf.float32, (1, 1024), 'mean_features')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        features_partial = model_module.create_encoder(inputs)
        _, fine = model_module.create_decoder \
            (features_partial, inputs, args.step_ratio, num_extract=512, mean_feature=mean_feature)

    dist1_fine, dist2_fine = tf_util.chamfer_distance(reconstruction, gt)
    if args.loss_type=='CD_P':
        total_loss = (tf.reduce_mean(tf.sqrt(dist1_fine)) + tf.reduce_mean(tf.sqrt(dist2_fine))) / 2
    elif args.loss_type=='CD_T':
        total_loss = tf.reduce_mean(dist1_fine) + tf.reduce_mean(dist2_fine)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    data_all=h5py.File(args.data_dir,'r')
    partial_all=data_all['incomplete_pcds'][()]
    complete_all=data_all['complete_pcds'][()]
    model_list = data_all['labels'][()].astype(int)

    saver.restore(sess, os.path.join(args.checkpoint))

    file_mean_feature = h5py.File(args.mean_features, 'r')
    mean_feature_data = file_mean_feature['mean_features'][()]
    file_mean_feature.close()

    total_time = 0
    cd_per_cat = {}

    total_cd = 0
    for i, model_id in enumerate(model_list):
        partial = partial_all[i]  #read_pcd(os.path.join(args.data_dir_novel, 'partial', '%s.pcd' % model_id))
        complete = complete_all[i]  # read_pcd(os.path.join(args.data_dir_novel, 'complete', '%s.pcd' % model_id))
        start = time.time()

        label=model_list[i]
        mean_feature_input=mean_feature_data[label].reshape(1, 1024)

        completion = sess.run(fine, feed_dict={inputs: [partial], is_training_pl: False,mean_feature:mean_feature_input})
        cd = sess.run(total_loss, feed_dict={reconstruction: completion, gt: [complete], is_training_pl: False})
        total_time += time.time() - start
        total_cd += cd

        category=objects[label]
        key_list = list(snc_synth_id_to_category.keys())
        val_list = list(snc_synth_id_to_category.values())
        synset_id=key_list[val_list.index(category)]

        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        cd_per_cat[synset_id].append(cd)
    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Chamfer distance per category')
    for synset_id in sorted(cd_per_cat.keys()):
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))

    sess.close()
    data_all.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--loss_type', default='CD_T')
    parser.add_argument('--data_dir', default='data/our_data/test_data.h5')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--num_gt_points', type=int, default=2048)
    parser.add_argument('--step_ratio', type=int, default=2)
    parser.add_argument('--model_type', default='cascaded_refinement_net')
    parser.add_argument('--mean_features', default='data/our_data/mean_feature.h5')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test(args)


