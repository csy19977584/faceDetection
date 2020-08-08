# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import random

# 分类问题的类数，fc层的输出单元个数
NUM_CLASSES = 5
# 更新中心的学习率
ALPHA = 0.5
#(0.6->0.5)
# center-loss的系数
LAMBDA = 0.0005


def center_loss(labels, features):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, 特征数目).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    alpha = ALPHA
    num_classes = NUM_CLASSES
    labels=K.argmax(labels, axis=-1)
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op]):
       c_loss = tf.nn.l2_loss(features - centers_batch)
        # 计算center-loss

    return c_loss



