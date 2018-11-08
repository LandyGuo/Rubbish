#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()



def dropBlock(inputs, block_size=5, keep_prob=0.9):
    """
    DropBlock implementation, refer to: https://arxiv.org/abs/1810.12890
    :param inputs: [batch, height, width, channel]
    :param block_size: size of drop-block size
    :param keep_prob: units to remain, drop 1-keep_prob units
    :return: same shape as inputs
    """
    b, h, w, c = inputs.get_shape().as_list()

    # calculate mask size
    side = (block_size-1)//2
    mask_height, mask_width = h-2*side, w-2*side

    # calculate gamma
    gamma = (1.-keep_prob)/block_size**2
    gamma *= h/(h-block_size+1.)
    gamma *= w/(w-block_size+1.)

    # generate Bernoulli distribution from gamma: 1 represents drop in mask
    sample_mask = tf.distributions.Bernoulli(probs=gamma).sample((b,mask_height,mask_width))

    # drop those centered  with block_size square where 1 appears
    # padding sample_mask, padding is not relevant with mask:
    # 1. padding sample_mask to origin input size: [(0,0), (side, side), (side,side)]
    # 2. padding origin input size with zeros for conv out as the same input_size or input_size+1
    # :[(0,0), (side+(block_size-1)/2, side+(block_size-1)/2), (side+(block_size-1)/2, side+(block_size-1)/2)]
    padding = (block_size + 2*side -1) // 2
    sample_mask = tf.pad(sample_mask, [[0,0], [padding, padding], [padding,padding]]) # batch x height x width
    sample_mask = tf.cast(tf.expand_dims(sample_mask, -1), tf.float32) # batch x height x width x 1
    filter_kernel = tf.ones((block_size, block_size, 1, 1), dtype=tf.float32) # block_size x block_size x 1 x 1
    conv_out = tf.nn.conv2d(sample_mask, filter_kernel, strides=[1,1,1,1], padding='VALID') # batch x height x width x 1

    # generate mask
    dropmask = tf.where(conv_out>0, tf.zeros_like(conv_out), tf.ones_like(conv_out)) # batch x height x width x 1
    dropmask = tf.cast(dropmask, tf.float32)

    # inputs: [b, h, w, c], 多个通道通过相同的处理
    masked_input = inputs * dropmask

    # normalize
    output =  masked_input * tf.size(dropmask, out_type=tf.float32) / tf.reduce_sum(dropmask)

    return output





if __name__=='__main__':
    inputs = tf.random_normal((1,22,22,1), dtype=tf.float32)
    print("origin inputs:", tf.reshape(inputs, (22,22)))
    outputs =  dropBlock(inputs)
    print("outputs:",  tf.reshape(outputs, (22,22)))
