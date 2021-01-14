import argparse
import importlib
import models
import os
import tensorflow as tf
import time
import sys
import h5py
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append('./utils')
sys.path.append(os.path.join(ROOT_DIR, 'pc_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='cascaded_refinement_net')
parser.add_argument('--lr_decay', default=False)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--h5_train',default='data/our_data/train_data.h5')  
parser.add_argument('--h5_val',default='data/our_data/valid_data.h5')  
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--gpu', default='1')
parser.add_argument('--allow_growth', action='store_false')
parser.add_argument('--step_ratio', type=int, default=2)
parser.add_argument('--num_gt_points', type=int, default=2048)
parser.add_argument('--mean_features', default='data/our_data/mean_feature.h5')
parser.add_argument('--mean_feature_size', type=int, default=1024)
parser.add_argument('--log_dir', default='log/me/test_')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--loss_type', type=str, default='CD_T')
parser.add_argument('--steps_per_print', type=int, default=1000)
parser.add_argument('--steps_per_eval', type=int, default=100)
parser.add_argument('--steps_per_save', type=int, default=2000)

parser.add_argument('--augment', action='store_true')
parser.add_argument('--pc_augm_scale', default=0.0, type=float,help='Training augmentation: Uniformly random scaling in [1/scale, scale]') # 1.1
parser.add_argument('--pc_augm_rot', default=0, type=int,help='Training augmentation: Bool, random rotation around z-axis')
parser.add_argument('--pc_augm_mirror_prob', default=0.0, type=float,help='Training augmentation: Probability of mirroring about x or y axes')
parser.add_argument('--pc_augm_jitter', default=0, type=int,help='Training augmentation: Bool, Gaussian jittering of all attributes')
parser.add_argument('--restore',  action='store_true') #    default=True  action='store_true'

parser.add_argument('--rec_weight', default=200.0, type=float)
parser.add_argument('--base_lr_d', type=float, default=0.00005)
parser.add_argument('--base_lr_g', type=float, default=0.0001)
parser.add_argument('--num_input_points', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=300)

parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
args = parser.parse_args()
if args.num_gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert (args.batch_size % args.num_gpus == 0)
DEVICE_BATCH_SIZE = args.batch_size / args.num_gpus

if args.pc_augm_scale>1:
    assert args.augment==True

args.log_dir = args.log_dir + str(args.step_ratio)

os.makedirs(args.log_dir,exist_ok=True)
os.makedirs(os.path.join(args.log_dir, 'plots'),exist_ok=True)
LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(args) + '\n')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        # for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000,20000,50000],[0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_input_points, 3), 'inputs')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')
    mean_feature = tf.placeholder(tf.float32, (args.batch_size, args.mean_feature_size), 'mean_features')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')

    file_train = h5py.File(args.h5_train, 'r')
    incomplete_pcds_train = file_train['incomplete_pcds'][()]
    complete_pcds_train = file_train['complete_pcds'][()]
    labels_train = file_train['labels'][()].astype(int)
    train_num = complete_pcds_train.shape[0]

    if args.lr_decay:
        lr_decay_step = int(train_num / args.batch_size * args.lr_decay_epochs)
        learning_rate_d = tf.where(
            tf.greater_equal(global_step//2, lr_decay_step),
            tf.train.exponential_decay(args.base_lr_d, global_step//2,
                                       lr_decay_step, args.lr_decay_rate, staircase=True),
            args.base_lr_d
        )
        learning_rate_d = tf.maximum(learning_rate_d, args.lr_clip)
        learning_rate_g = tf.where(
            tf.greater_equal(global_step//2, lr_decay_step),
            tf.train.exponential_decay(args.base_lr_g, global_step//2,
                                       lr_decay_step, args.lr_decay_rate, staircase=True),
            args.base_lr_g
        )
        learning_rate_g = tf.maximum(learning_rate_g, args.lr_clip)
    else:
        learning_rate_g=tf.constant(args.base_lr_g, name='lr_g')
        learning_rate_d = tf.constant(args.base_lr_d, name='lr_d')

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=0.9)
        D_optimizers = tf.train.AdamOptimizer(learning_rate_d, beta1=0.5)

    tower_grads_g = []
    tower_grads_d = []
    coarse_gpu = []
    fine_gpu = []
    total_dis_loss_gpu=[]
    errG_loss_gpu=[]
    total_gen_loss_gpu=[]
    total_loss_rec_gpu=[]

    for i in range(args.num_gpus):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:

                inputs_pl_batch = tf.slice(inputs_pl, [int(i * DEVICE_BATCH_SIZE), 0, 0],[int(DEVICE_BATCH_SIZE), -1, -1])
                gt_pl_batch = tf.slice(gt_pl, [int(i * DEVICE_BATCH_SIZE), 0, 0], [int(DEVICE_BATCH_SIZE), -1, -1])
                mean_feature_batch = tf.slice(mean_feature, [int(i * DEVICE_BATCH_SIZE), 0],[int(DEVICE_BATCH_SIZE), -1])

                ### generator ###
                with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                    features_partial_batch = model_module.create_encoder(inputs_pl_batch)
                    coarse_batch, fine_batch = model_module.create_decoder \
                        (features_partial_batch, inputs_pl_batch, args.step_ratio, num_extract=512, mean_feature=mean_feature_batch)

                assert fine_batch.get_shape()[1].value == args.step_ratio * 1024

                dist1_fine, dist2_fine = tf_util.chamfer_distance(fine_batch, gt_pl_batch)
                dist1_coarse, dist2_coarse = tf_util.chamfer_distance(coarse_batch, gt_pl_batch)

                total_loss_fine = (tf.reduce_mean(tf.sqrt(dist1_fine)) + tf.reduce_mean(tf.sqrt(dist2_fine))) / 2
                total_loss_coarse = (tf.reduce_mean(tf.sqrt(dist1_coarse)) + tf.reduce_mean(tf.sqrt(dist2_coarse))) / 2
                total_loss_rec_batch = alpha * total_loss_fine + total_loss_coarse

                ### discriminator ###
                with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                    d_fake = model_module.patch_dection(fine_batch[:,0:2048,:], divide_ratio=2)
                    d_real = model_module.patch_dection(gt_pl_batch[:,0:2048,:], divide_ratio=2)
                d_loss_real = tf.reduce_mean((d_real - 1) ** 2)
                d_loss_fake = tf.reduce_mean(d_fake ** 2)
                errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
                errG_loss_batch = tf.reduce_mean((d_fake - 1) ** 2)

                t_vars = tf.global_variables()
                gen_tvars = [var for var in t_vars if var.name.startswith("generator")]  # (var.name.startswith("ae") or var.name.startswith("agg") or var.name.startswith("decoder"))
                dis_tvars = [var for var in t_vars if var.name.startswith("discriminator")]
                clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_tvars]

                total_gen_loss_batch = errG_loss_batch + total_loss_rec_batch * args.rec_weight
                total_dis_loss_batch = errD_loss_batch

                # Calculate the gradients for the batch of data on this tower.
                grads_g = G_optimizers.compute_gradients(total_gen_loss_batch, var_list=gen_tvars)
                grads_d = D_optimizers.compute_gradients(total_dis_loss_batch, var_list=dis_tvars)

                # Keep track of the gradients across all towers.
                tower_grads_g.append(grads_g)
                tower_grads_d.append(grads_d)

                coarse_gpu.append(coarse_batch)
                fine_gpu.append(fine_batch)
                total_dis_loss_gpu.append(total_dis_loss_batch)
                errG_loss_gpu.append(errG_loss_batch)
                total_gen_loss_gpu.append(total_gen_loss_batch)
                total_loss_rec_gpu.append(total_loss_rec_batch)

    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    train_G = G_optimizers.apply_gradients(grads_g, global_step=global_step)
    train_D = D_optimizers.apply_gradients(grads_d, global_step=global_step)

    fine = tf.concat(fine_gpu, 0)
    total_dis_loss = tf.reduce_mean(total_dis_loss_gpu)
    errG_loss = tf.reduce_mean(errG_loss_gpu)
    total_gen_loss = tf.reduce_mean(total_gen_loss_gpu)
    total_loss_rec = tf.reduce_mean(total_loss_rec_gpu)

    dist1_eval, dist2_eval = tf_util.chamfer_distance(fine, gt_pl)

    file_val = h5py.File(args.h5_val, 'r')
    incomplete_pcds_val = file_val['incomplete_pcds'][()]
    complete_pcds_val = file_val['complete_pcds'][()]
    labels_val = file_val['labels'][()]
    file_val.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    file_mean_feature = h5py.File(args.mean_features, 'r')
    mean_feature_data = file_mean_feature['mean_features'][()]
    file_mean_feature.close()

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        print('load model', args.log_dir)
    else:
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))
        os.system('cp train.py %s' % args.log_dir)

    train_ids = np.arange(incomplete_pcds_train.shape[0])
    train_num = incomplete_pcds_train.shape[0]

    init_step = sess.run(global_step//2)
    epoch = init_step * args.batch_size // train_num + 1
    print('init_step:%d,' % init_step, 'epoch:%d' % epoch,'training data number:%d'%train_num)

    best_loss=10

    num_batches = train_num // args.batch_size
    for ep_cnt in range(epoch, args.max_epoch+1):

        np.random.shuffle(train_ids)
        for batch_idx in range(num_batches):
            init_step+=1

            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, train_num)
            ids_train = train_ids[start_idx:end_idx]
            batch_data = incomplete_pcds_train[ids_train]
            batch_gt = complete_pcds_train[ids_train]

            if args.augment:
                for k in range(end_idx - start_idx):
                    batch_gt[k], batch_data[k] = tf_util.augment_cloud([batch_gt[k], batch_data[k]], args)

            labels = labels_train[ids_train].astype(int)
            mean_feature_input = mean_feature_data[labels]

            feed_dict = {inputs_pl: batch_data, gt_pl: batch_gt, is_training_pl: True,mean_feature: mean_feature_input}
            _,_, loss_dis = sess.run([train_D,clip_D, total_dis_loss], feed_dict=feed_dict)
            _, loss_gen, rec_loss,errg_loss = sess.run([train_G, total_gen_loss, total_loss_rec, errG_loss],feed_dict=feed_dict)

            if init_step % args.steps_per_print == 0:
                log_string('epoch %d  step %d dis_loss %.8f total_gen_loss %.8f rec_loss %.8f gen_loss %.8f'%(ep_cnt,init_step,loss_dis,loss_gen,rec_loss,errg_loss))

            if init_step % args.steps_per_eval == 0:
                total_loss = 0
                sess.run(tf.local_variables_initializer())
                batch_data = np.zeros((args.batch_size, incomplete_pcds_val[0].shape[0], 3), 'f')
                batch_gt = np.zeros((args.batch_size, args.num_gt_points, 3), 'f')
                labels=np.zeros((args.batch_size,), dtype=np.int32)

                for batch_idx_eval in range(0, incomplete_pcds_val.shape[0], args.batch_size):
                    start_idx = batch_idx_eval
                    end_idx = min(start_idx+ args.batch_size,incomplete_pcds_val.shape[0])

                    batch_data[0:end_idx - start_idx] = incomplete_pcds_val[start_idx:end_idx]
                    batch_gt[0:end_idx - start_idx] = complete_pcds_val[start_idx:end_idx]
                    labels[0:end_idx - start_idx] = labels_val[start_idx:end_idx]

                    mean_feature_input_eval = mean_feature_data[labels]

                    feed_dict = {inputs_pl: batch_data, gt_pl: batch_gt, is_training_pl: False, mean_feature: mean_feature_input_eval}
                    dist1_out , dist2_out = sess.run([dist1_eval, dist2_eval], feed_dict=feed_dict)
                    if args.loss_type == 'CD_T':
                        total_loss += np.mean(dist1_out[0:end_idx - start_idx]) * (end_idx - start_idx) \
                                      + np.mean(dist2_out[0:end_idx - start_idx]) * (end_idx - start_idx)
                    elif args.loss_type == 'CD_P':
                        total_loss += (np.mean(np.sqrt(dist1_out[0:end_idx - start_idx])) * (end_idx - start_idx) \
                                       + np.mean(np.sqrt(dist2_out[0:end_idx - start_idx])) * (end_idx - start_idx)) / 2

                if total_loss / incomplete_pcds_val.shape[0] < best_loss:
                    best_loss = total_loss / incomplete_pcds_val.shape[0]
                    saver.save(sess, os.path.join(args.log_dir, 'model'), init_step)

                log_string('epoch %d  step %d  loss %.8f best_loss %.8f' %(ep_cnt, init_step, total_loss / incomplete_pcds_val.shape[0], best_loss))

            if init_step % args.steps_per_save == 0:
                saver.save(sess, os.path.join(args.log_dir, 'model'), init_step)

    file_train.close()
    sess.close()

if __name__ == '__main__':
    train(args)
