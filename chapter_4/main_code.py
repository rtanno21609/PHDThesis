import tensorflow as tf

# inputs and annotators' labels
images = tf.placeholder(tf.float32, (None, image_shape))
labels = tf.placeholder(tf.int32,(None, num_annotators, num_classes))

# classifier for estimating true label distribution
logits = classifier(images)

# confusion matrices of annotators
confusion_matrices = confusion_matrix_estimators(num_annotators, num_classes)

# loss function
# 1. weighted cross-entropy
weighted_cross_entropy = cross_entropy_over_annotators(labels, logits, confusion_matrices)

# 2. trace of confusion matrices:
trace_norm = tf.reduce_mean(tf.trace(confusion_matrices))

# final loss (eq.(4))
total_loss = weighted_cross_entropy + scale * trace_norm