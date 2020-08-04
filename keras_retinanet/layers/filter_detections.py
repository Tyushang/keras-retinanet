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

    n_anchor, n_class = tf.unstack(tf.shape(classification), name='tyu_fd_unstack128')
    # add dummy box or score at the end. invalid indices will point to the dummy one.
    # shape: [n_anchor + 1, 4]
    boxes_with_dummy  = tf.concat([boxes, tf.zeros(shape=(1, 4), dtype=boxes.dtype)], axis=0, name='tyu_fd_concat131')
    # shape: [n_anchor + 1, n_class]
    scores_with_dummy = tf.concat([classification, float('-inf') * tf.ones((1, n_class), classification.dtype)], axis=0, name='tyu_fd_concat133')
    # shape: [n_anchor + 1, ...]
    # other_with_dummy  = [tf.concat([x, [tf.zeros_like(other[0], dtype=x.dtype)]], axis=0) for x in other]

    def _sort_by_scores(scores, others, output_size, use_nms=False):
        """sort by scores, dim0_size of scores and *others must be identical cause all will be sorted."""
        dummy_index = tf.shape(scores)[0] - 1
        if use_nms:
            boxes = others[0]
            indices_padded, n_valid = backend.non_max_suppression_padded(boxes, scores,
                                                                         max_output_size=output_size,
                                                                         iou_threshold=nms_threshold,
                                                                         score_threshold=score_threshold,
                                                                         pad_to_max_output_size=True)
            # replace invalid indices to 'n_anchor' which point to dummy one(last one).
            indices_padded = tf.where(tf.range(max_detections) < n_valid, indices_padded, dummy_index, name='tyu_fd_where149')
        else:
            topk_scores, indices_padded = tf.nn.top_k(scores, k=output_size)
            # replace invalid indices to 'n_anchor' which point to dummy one(last one).
            indices_padded = tf.where(topk_scores > score_threshold, indices_padded, dummy_index, name='tyu_fd_153')

        return tuple([tf.gather(x, indices_padded) for x in [scores, *others]])

    def _pre_exclude(args):
        """pre-exclude those with low scores, using top-k method, to reduce memory usage."""
        scores, *others = args
        return _sort_by_scores(scores, others, output_size=(max_detections * 2), use_nms=False)

    def _sort_by_1st_arg_using_topk(args):
        scores, *others = args
        return _sort_by_scores(scores, others, output_size=max_detections, use_nms=False)

    def _sort_by_1st_arg_using_nms(args):
        scores, *others = args
        return _sort_by_scores(scores, others, output_size=max_detections, use_nms=True)

    if class_specific_filter:
        # scores_adapt shape: [n_class, n_anchor + 1], labels shape: [n_class, n_anchor]
        scores_adapt = tf.transpose(scores_with_dummy)
        labels       = tf.broadcast_to(tf.range(n_class)[:, tf.newaxis], shape=(n_class, n_anchor))
    else:
        # scores_adapt shape: [      1, n_anchor + 1], labels shape: [      1, n_anchor]
        scores_adapt = tf.reshape(tf.reduce_max(scores_with_dummy, axis=1), shape=(1, -1))
        labels       = tf.reshape(tf.argmax(classification, axis=1, output_type=tf.int32), shape=(1, -1))
    # dim0_size: 1 or n_class
    dim0_size = tf.shape(scores_adapt)[0]
    # shape: [dim0_size, n_anchor + 1, ...] after-same. add dummy label at the end.
    labels_adapt = tf.concat([labels, -1 * tf.ones((dim0_size, 1), dtype=labels.dtype)], axis=1)
    boxes_adapt  = tf.broadcast_to(boxes_with_dummy[tf.newaxis, ...], shape=(dim0_size, *tf.unstack(tf.shape(boxes_with_dummy))))
    # other_adapt  = [tf.broadcast_to(x[tf.newaxis, ...], shape=(dim0_size, *tf.unstack(tf.shape(x)))) for x in other_with_dummy]

    func  = _sort_by_1st_arg_using_nms if nms else _sort_by_1st_arg_using_topk
    elems = (scores_adapt, boxes_adapt, labels_adapt, )  # , *other_adapt)
    # pre-exclude to reduce memory usage.
    res   = tf.map_fn(_pre_exclude, elems=elems, dtype=tuple([x.dtype for x in elems]), name='tyu_fd_map174')
    # shape: [dim0_size, max_detections, ...]
    res   = tf.map_fn(func, elems=elems, dtype=tuple([x.dtype for x in res]), name='tyu_fd_map174')
    # shape: [dim0_size * max_detections, ...], flatten first 2 dimensions.
    res   = [tf.reshape(x, (tf.shape(x)[0] * tf.shape(x)[1], *tf.unstack(tf.shape(x)[2: ]))) for x in res]

    if class_specific_filter:
        # shape: [max_detections, ...]
        res = _sort_by_1st_arg_using_topk(res)

    scores_res, boxes_res, labels_res, *other_res = res
    scores_res.set_shape(shape=(max_detections, ))
    boxes_res.set_shape(shape=(max_detections, 4))
    labels_res.set_shape(shape=(max_detections, ))

    return tuple([boxes_res, scores_res, labels_res]) # + other_res)


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
        boxes, classification, *other = inputs
        # shape: [B, N, 4]
        boxes = tf.identity(boxes, name='tyu_fd_id227')
        # shape: [B, N]
        classification = tf.identity(classification, name='tyu_fd_id228')

        # wrap nms with our parameters
        def _filter_detections(args):
            b, c = args
            b = tf.identity(b, name='tyu_fd_id233')
            c = tf.identity(c, name='tyu_fd_id234')
            return filter_detections_tpu(
                b, c, None,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=(boxes, classification),
            dtype=(tf.float32, tf.float32, tf.int32), # tuple([] + [o.dtype for o in other]),
            # parallel_iterations=self.parallel_iterations,
            name='tyu_fd_map_fn246'
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
