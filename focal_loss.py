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
