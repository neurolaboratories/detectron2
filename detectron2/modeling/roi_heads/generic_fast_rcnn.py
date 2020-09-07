import numpy as np
import torch

from detectron2.layers import batched_nms
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, Instances, pairwise_iou

__all__ = ["GenericFastRCNNOutputLayers"]
"""
Shape shorthand in this module:
    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.
Naming convention:
    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).
    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).
    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.
    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.
    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class GenericFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_shape):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
        """
        super().__init__(cfg, input_shape)

        # Threshold to filter predictions results
        self.class_score_thresh_test = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        # NMS threshold for prediction results.
        self.class_nms_thresh_test = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        # Return detections with a objectness score exceeding this threshold.
        self.objectness_score_thresh_test = cfg.MODEL.ROI_HEADS.OBJECTNESS_SCORE_THRESH_TEST
        # The threshold to use for box non-maximum suppression when applied
        # to objects. Value in [0, 1].
        self.objectness_nms_thresh_test = cfg.MODEL.ROI_HEADS.OBJECTNESS_NMS_THRESH_TEST
        # Number of top predictions to produce per image.
        self.topk_per_image_test = cfg.TEST.DETECTIONS_PER_IMAGE

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `generic_inference`.
            list[Tensor]: same as `generic_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return self._generic_inference(
            boxes=boxes, scores=scores, proposals=proposals, image_shapes=image_shapes
        )

    def _generic_inference(self, boxes, scores, proposals, image_shapes):
        """
        Call `generic_inference_single_image` for all images.
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific
                or class-agnostic boxes for each image. Element i has shape
                (Ri, K * 4) if doing class-specific regression, or (Ri, 4) if
                doing class-agnostic regression, where Ri is the number of
                predicted objects for image i. This is compatible with the
                output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores
                for each image. Element i has shape (Ri, K + 1), where Ri is the
                number of predicted objects for image i. Compatible with the
                output of :meth:`FastRCNNOutputLayers.predict_probs`.
            proposals (list[Instances]): proposals that match the features
                    that were used to compute predictions.
            image_shapes (list[tuple]): A list of (width, height) tuples for
                each image in the batch.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each
                image in the batch, that stores the topk most confidence
                detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N,
                each element indicates the corresponding boxes/scores index in
                [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self._generic_inference_single_image(
                boxes_per_image, scores_per_image, proposals_per_image, image_shape
            )
            for scores_per_image, proposals_per_image, boxes_per_image, image_shape in zip(
                scores, proposals, boxes, image_shapes
            )
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def _generic_inference_single_image(self, boxes, scores, proposals, image_shape):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        Args:
            Same as `generic_inference`, but with boxes, scores, proposals and image shapes
            per image.
        Returns:
            Same as `generic_inference`, but for only one image.
        """

        # Filter invalid choices
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)

        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        scores = scores[:, :-1]
        # New class as generic, gets the highest ID + 1
        generic_idx = scores.shape[1]

        class_boxes, class_scores, class_inds = self._get_class_predictions(
            boxes=boxes, scores=scores, image_shape=image_shape
        )

        generic_boxes, generic_scores, generic_inds = self._get_generic_predictions(
            proposals=proposals,
            class_boxes=class_boxes,
            class_scores=class_scores,
            class_inds=class_inds,
            generic_idx=generic_idx,
        )

        out_boxes = torch.cat((class_boxes, generic_boxes), 0)
        out_scores = torch.cat((class_scores, generic_scores), 0)
        out_ids = torch.cat((class_inds, generic_inds), 0)

        result = Instances(image_shape)
        result.pred_boxes = Boxes(out_boxes)
        result.scores = out_scores
        result.pred_classes = out_ids[:, 1]
        return result, out_ids[:, 0]

    def _get_class_predictions(self, boxes, scores, image_shape):

        num_bbox_reg_classes = boxes.shape[1] // 4

        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # Filter results based on detection scores
        filter_mask = scores > self.class_score_thresh_test

        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        class_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[class_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # Apply per-class NMS
        keep_class = batched_nms(boxes, scores, class_inds[:, 1], self.class_nms_thresh_test)
        if self.topk_per_image_test >= 0:
            keep_class = keep_class[: self.topk_per_image_test]

        boxes, scores, class_inds = boxes[keep_class], scores[keep_class], class_inds[keep_class]

        return boxes, scores, class_inds

    def _get_generic_predictions(
        self,
        proposals: Instances,
        class_boxes: torch.FloatTensor,
        class_scores: torch.FloatTensor,
        class_inds: torch.FloatTensor,
        generic_idx: int,
    ) -> (torch.FloatTensor, torch.FloatTensor, torch.IntTensor):
        """Select generic predictions with objectness score larger than a threshold and that don't match class objects

        Args:
            proposals (Instances): Predicted generic instances
            class_boxes (torch.FloatTensor): Bbox predictions of class instance
            class_scores (torch.FloatTensor): Scores of class instances
            class_inds (torch.FloatTensor): Class indices of class instances
            generic_idx (int): Generic class id

        Returns:
            (torch.FloatTensor, torch.FloatTensor, torch.IntTensor): Returns a tuple formed of a 1D tensor of generic bboxes, 
            a 1D tensor of generic scores and a 2D tensor formed of the generic prediction index and the class id it represents
        """
        #####
        # Per object
        objectness = proposals.objectness_logits.reshape((proposals.objectness_logits.shape[0], 1))

        obj_boxes = proposals.proposal_boxes.tensor

        # Filter by objectness threshold
        filter_object_mask = objectness > self.objectness_score_thresh_test

        filter_obj_inds = filter_object_mask.nonzero()
        obj_boxes = obj_boxes[filter_obj_inds[:, 0]]

        # Filter generic objects that overlap with class predictions
        generic_mask = self._find_generic_objects_suppression_mask(
            class_boxes, obj_boxes, self.objectness_nms_thresh_test
        )

        objectness = objectness[filter_object_mask]

        generic_boxes = obj_boxes[generic_mask]
        generic_inds = filter_obj_inds[:][generic_mask]
        generic_scores = objectness[generic_mask]

        # Attribute generic id to selected predictions
        generic_inds[:, 1] = generic_idx

        # Apply NMS to generic predictions
        nms_filtered = batched_nms(
            generic_boxes, generic_scores, generic_inds[:, 1], self.objectness_nms_thresh_test
        )

        generic_boxes = generic_boxes[nms_filtered]
        generic_inds = generic_inds[:][nms_filtered]
        generic_scores = generic_scores[nms_filtered]

        # Keep top detections - detected classes have priority
        if self.topk_per_image_test >= 0:
            remaining_objects = self.topk_per_image_test - len(class_boxes)
            sorted_generic = np.argsort(generic_scores)
            sorted_generic = sorted_generic[:remaining_objects]

            generic_boxes = generic_boxes[sorted_generic]
            generic_inds = generic_inds[sorted_generic]
            generic_scores = generic_scores[sorted_generic]

        return generic_boxes, generic_scores, generic_inds

    def _find_generic_objects_suppression_mask(
        self,
        class_boxes: torch.FloatTensor,
        generic_boxes: torch.FloatTensor,
        threshold_generic_iou: float = 0.7,
    ) -> np.ndarray:
        """Find the mask for all generic object instances that have IoU < threshold 
        than all of the class instances

        Args:
            class_boxes (torch.FloatTensor): [N, 4] tensor with all bbox coordinates for class instances
            generic_boxes (torch.FloatTensor): [M, 4] tensor with all bbox coordinates for generic instances
            threshold_generic_iou (float, optional): Overlap threshold at which to cutoff generic instances. Defaults to 0.7.

        Returns:
            np.ndarray: Mask of boolean values with True at position indexes that should be kept.
        """
        pair_ious = pairwise_iou(Boxes(generic_boxes), Boxes(class_boxes))
        mask_generic_bboxes = np.zeros((len(generic_boxes)), dtype=bool)
        for index, matchings_to_classes in enumerate(pair_ious):
            mask_generic_bboxes[index] = (
                max(matchings_to_classes, default=0.0) < threshold_generic_iou
            )
        return mask_generic_bboxes
