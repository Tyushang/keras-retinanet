"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from .. import backend


def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = backend.where(tf.keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = tf.keras.backend.gather(scores, indices)[:, 0]
            # perform NMS
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            # filter indices based on NMS
            indices = tf.keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = backend.gather_nd(labels, indices)
        indices = tf.keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * backend.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = tf.keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = tf.keras.backend.max(classification, axis    = 1)
        labels  = tf.keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes               = tf.keras.backend.gather(boxes, indices)
    labels              = tf.keras.backend.gather(labels, top_indices)
    other_              = [tf.keras.backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = tf.keras.backend.cast(labels, 'int32')
    other_   = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(tf.keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


def filter_detections_tpu(
    boxes,
    classification,
    other                 = None,
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    """TPU version."""
    if other is None:
        other = []

    n_anchor, n_class = tf.unstack(tf.shape(classification))
    # add dummy box or score at the end. invalid indices returned by non_max_suppression_padded will point dummy one.
    # shape: [n_anchor + 1, 4]
    boxes_concat  = tf.concat([boxes, tf.zeros(shape=(1, 4), dtype=boxes.dtype)], axis=0)
    # shape: [n_anchor + 1, n_class]
    scores_concat = tf.concat([classification, float('-inf') * tf.ones(shape=(1, n_class))], axis=0)
    # shape: [d0 + 1, d1, ...]
    other_concat = [tf.concat([x, [tf.zeros_like(other[0])]], axis=0) for x in other]

    def _handle_one_entry(scores):
        # scores shape: [n_anchor + 1, ]
        if nms:
            indices_padded, n_valid = backend.non_max_suppression_padded(boxes_concat,
                                                                         scores,
                                                                         max_output_size=max_detections,
                                                                         iou_threshold=nms_threshold,
                                                                         score_threshold=score_threshold,
                                                                         pad_to_max_output_size=True)
            # replace invalid indices to 'n_anchor' which point to dummy one.
            indices_padded = tf.where(tf.range(max_detections) < n_valid, indices_padded, n_anchor)
        else:
            _, indices_padded = tf.nn.top_k(scores, max_detections)

        return indices_padded

    if class_specific_filter:
        # shape: [n_class, n_anchors + 1]
        adapted_scores    = tf.transpose(scores_concat)
        _, adapted_labels = tf.meshgrid(tf.range(n_anchor + 1), tf.range(n_class))
    else:
        # shape: [1, n_anchor + 1], after-same.
        adapted_scores = tf.reshape(tf.reduce_max(scores_concat, axis=1), shape=(1, -1))
        adapted_labels = tf.reshape(tf.argmax(scores_concat, axis=1, output_type=tf.int32), shape=(1, -1))

    # add dummy label at the end.
    adapted_labels = tf.where(tf.reshape(tf.range(n_anchor + 1), (1, n_anchor + 1)) < n_anchor, adapted_labels, -1)
    # shape: [n_class or 1, max_detections], after-same.
    nms_indices = tf.map_fn(_handle_one_entry, elems=adapted_scores, dtype=tf.int32)
    _, iy       = tf.meshgrid(tf.range(max_detections), tf.range(tf.shape(nms_indices)[0]))
    nms_scores  = tf.gather_nd(adapted_scores, tf.stack([iy, nms_indices], axis=-1))
    nms_labels  = tf.gather_nd(adapted_labels, tf.stack([iy, nms_indices], axis=-1))
    # shape: [max_detections, ]
    topk_scores, topk_indices = tf.nn.top_k(tf.reshape(nms_scores, shape=(-1, )), k=max_detections)
    # top-k indices to anchor indices
    anchor_indices  = tf.gather(tf.reshape(nms_indices, shape=(-1, )), topk_indices)
    # lookup nms_labels to get topk_labels.
    topk_labels = tf.gather(tf.reshape(nms_labels, shape=(-1, )), topk_indices)
    # lookup boxes_concat/other_concat to get topk_boxes/topk_other.
    topk_boxes  = tf.gather(boxes_concat, anchor_indices)
    topk_other  = [tf.gather(x, anchor_indices) for x in other_concat]

    tf.print(topk_labels)

    return [topk_boxes, topk_scores, topk_labels] + topk_other


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections_tpu(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[tf.keras.backend.floatx(), tf.keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config
