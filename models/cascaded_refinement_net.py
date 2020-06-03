import tensorflow as tf
import tf_util
from tf_sampling import farthest_point_sample, gather_point
import math

def create_encoder(inputs,embed_size=1024):
    inputs_new=inputs
    features = tf_util.mlp_conv(inputs_new, [128, 256], bn=None, bn_params=None,name='encoder_0')
    features_global = tf.reduce_max(features, axis=1, keepdims=True, name='maxpool_0')
    features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs_new)[1], 1])], axis=2)
    features = tf_util.mlp_conv(features, [512, embed_size], bn=None, bn_params=None,name='encoder_1')
    features = tf.reduce_max(features, axis=1, name='maxpool_1')
    return features

def symmetric_sample(points, num):
    p1_idx = farthest_point_sample(num, points)
    input_fps = gather_point(points, p1_idx)
    input_fps_flip = tf.concat(
        [tf.expand_dims(input_fps[:, :, 0], axis=2), tf.expand_dims(input_fps[:, :, 1], axis=2),
         tf.expand_dims(-input_fps[:, :, 2], axis=2)], axis=2)
    input_fps = tf.concat([input_fps, input_fps_flip], 1)
    return input_fps

def contract_expand_operation(inputs,up_ratio):
    net = inputs
    net = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
    net = tf.transpose(net, [0, 2, 1, 3])

    net = tf_util.conv2d(net,
                       64,
                       [1, up_ratio],
                       scope='down_conv1',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net = tf_util.conv2d(net,
                       128,
                       [1, 1],
                       scope='down_conv2',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net = tf.reshape(net, [tf.shape(net)[0], -1, up_ratio,64])
    net = tf_util.conv2d(net,
                       64,
                       [1, 1],
                       scope='down_conv3',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net=tf.reshape(net,[tf.shape(net)[0], -1, 64])
    return net

def create_decoder(code,inputs,step_ratio,num_extract=512,mean_feature=None):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

        level0 = tf_util.mlp(code, [1024, 1024, 512 * 3], bn=None, bn_params=None, name='coarse')  # ,name='coarse'
        level0 = tf.tanh(level0)
        level0 = tf.reshape(level0, [-1, 512, 3])
        coarse = level0

        input_fps = symmetric_sample(inputs, int(num_extract/2))
        level0 = tf.concat([input_fps, level0], 1)
        if num_extract>512:
            level0=gather_point(level0,farthest_point_sample(1024,level0))

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = tf_util.gen_grid_up(2 ** (i + 1))
            grid = tf.expand_dims(grid, 0)
            grid_feat = tf.tile(grid, [level0.shape[0], 1024, 1])
            point_feat = tf.tile(tf.expand_dims(level0, 2), [1, 1, 2, 1])
            point_feat = tf.reshape(point_feat, [-1, num_fine, 3])
            global_feat = tf.tile(tf.expand_dims(code, 1), [1, num_fine, 1])

            mean_feature_use=tf.contrib.layers.fully_connected(mean_feature,128,activation_fn=tf.nn.relu,scope='mean_fc')
            mean_feature_use = tf.expand_dims(mean_feature_use, 1)
            mean_feature_use = tf.tile(mean_feature_use, [1, num_fine, 1])
            feat = tf.concat([grid_feat, point_feat, global_feat,mean_feature_use], axis=2)

            feat1=tf_util.mlp_conv(feat, [128, 64], bn=None, bn_params=None,name='up_branch')
            feat1=tf.nn.relu(feat1)
            feat2=contract_expand_operation(feat1, 2)
            feat=feat1+feat2

            fine = tf_util.mlp_conv(feat, [512, 512, 3], bn=None, bn_params=None, name='fine') + point_feat

            level0 = fine
        return coarse,fine

def patch_dection(point_cloud, divide_ratio=1):
    with tf.variable_scope('patch', reuse=tf.AUTO_REUSE):
        l0_xyz = point_cloud
        l0_points = None
        num_point = point_cloud.get_shape()[1].value
        l1_xyz, l1_points = tf_util.pointnet_sa_module_msg(l0_xyz, l0_points, int(num_point/8), [0.1, 0.2, 0.4], [16, 32, 128],
                                                   [[32//divide_ratio, 32//divide_ratio, 64//divide_ratio], \
                                                    [64//divide_ratio, 64//divide_ratio, 128//divide_ratio], \
                                                    [64//divide_ratio, 96//divide_ratio, 128//divide_ratio]],
                                                   scope='layer1', use_nchw=False)
        patch_values=tf_util.mlp_conv(l1_points, [1], bn=None, bn_params=None,name='patch')
        return  patch_values