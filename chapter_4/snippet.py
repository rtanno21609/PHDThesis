import numpy as np
import tensorflow as tf

def cross_entropy_over_annotators(labels, logits, confusion_matrices):
    """ Cross entropy between noisy labels from multiple annotators and their confusion matrix models.
    Args:
        labels: One-hot representation of labels from multiple annotators. 
            tf.Tensor of size [batch, num_annotators, num_classes]. Missing labels are assumed to be
            represented as zero vectors.
        logits: Logits from the classifier. tf.Tensor of size [batch, num_classes]
        confusion_matrices: Confusion matrices of annotators. tf.Tensor of size 
            [num annotators, num_classes, num_classes]. The (i, j) th element of the confusion matrix 
            for annotator a denotes the probability P(label_annotator_a = j|label_true = i).
    Returns:
         The average cross-entropy across annotators and image examples.
    """
    # Treat one-hot labels as probability vectors
    labels = tf.cast(labels, dtype=tf.float32)

    # Sequentially compute the loss for each annotator
    losses_all_annotators = []
    for idx, labels_annotator in enumerate(tf.unstack(labels, axis=1)):
        loss = sparse_confusion_matrix_softmax_cross_entropy(
            labels=labels_annotator,
            logits=logits,
            confusion_matrix=confusion_matrices[idx, :, :],
        )
        losses_all_annotators.append(loss)

    # Stack them into a tensor of size (batch, num_annotators)
    losses_all_annotators = tf.stack(losses_all_annotators, axis=1)

    # Filter out annotator networks with no labels. This allows you train
    # annotator networks only when the labels are available.
    has_labels = tf.reduce_sum(labels, axis=2)
    losses_all_annotators = losses_all_annotators * has_labels
    return tf.reduce_mean(tf.reduce_sum(losses_all_annotators, axis=1))

def sparse_confusion_matrix_softmax_cross_entropy(labels, logits, confusion_matrix):
    """Cross entropy between noisy labels and confusion matrix based model for a single annotator.
    Args:
        labels: One-hot representation of labels. Tensor of size [batch, num_classes].
        logits: Logits from the classifier. Tensor of size [batch, num_classes]
        confusion_matrix: Confusion matrix of the annotator. Tensor of size [num_classes, num_classes].
    Returns:
         The average cross-entropy across annotators for image examples
         Returns a `Tensor` of size [batch_size].
    """
    # get the predicted label distribution
    preds_true = tf.nn.softmax(logits)

    # Map label distribution into annotator label distribution by
    # multiplying it by its confusion matrix.
    preds_annotator = tf.matmul(preds_true, confusion_matrix)
    
    # cross entropy
    preds_clipped = tf.clip_by_value(preds_annotator, 1e-10, 0.9999999)
    cross_entropy = tf.reduce_sum(-labels * tf.log(preds_clipped), axis=-1)
    return cross_entropy