# -*- coding: utf-8 -*-
import tensorflow as tf
# binary focal loss
def focal_loss(labels, logits, name, alpha=0.25, gamma=2.):
    with tf.name_scope("focal_loss"):
        # labels: [N]
        # logits: [N]
        labels = tf.cast(labels, tf.int32)
        logits = tf.cast(logits, tf.float32)

        logits = tf.sigmoid(logits)

        pt_1 = tf.where(tf.equal(labels, tf.ones_like(labels, dtype=tf.int32)), logits, tf.ones_like(labels, dtype=tf.float32))
        pt_0 = tf.where(tf.equal(labels, tf.zeros_like(labels, dtype=tf.int32)), logits, tf.zeros_like(labels, dtype=tf.float32))

        # clip for stablize
        eps = 1e-6
        pt_1 = tf.clip_by_value(pt_1, eps, 1.-eps)
        pt_0 = tf.clip_by_value(pt_0, eps, 1.-eps)

        counts = tf.cast(tf.shape(labels)[0], tf.float32)

        loss = -tf.reduce_sum(alpha*tf.pow(1.-pt_1, gamma)*tf.log(pt_1)+(1-alpha)*tf.pow(pt_0, gamma)*tf.log(1.-pt_0))
        return tf.div(loss, counts, name=name)
    
    
    


"""
该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
"""

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L

if __name__ == '__main__':
    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
    labels=tf.Variable([0,1,0,0,1])
    loss1=focal_loss_sigmoid(labels=labels,logits=logits)

    logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
    labels2=tf.Variable([1,0,2,3,1])
    loss2=focal_loss_softmax(labels==labels2,logits=logits2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(loss1)
        print sess.run(loss2)
