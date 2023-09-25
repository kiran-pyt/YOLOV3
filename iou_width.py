import torch

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    # Calculate the intersection of the boxes' widths and heights
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )

    # Calculate the union of the boxes' areas
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )

    # Calculate the Intersection over Union (IoU)
    iou = intersection / union

    return iou

# Example:
# Let's assume we have two sets of bounding boxes represented by their widths and heights.
# We'll calculate the IoU between each pair of bounding boxes.

# Bounding boxes represented as [width, height]
boxes1 = torch.tensor([[4.0, 3.0], [2.0, 2.0], [5.0, 5.0]])
boxes2 = torch.tensor([[3.0, 2.0], [3.0, 4.0], [6.0, 4.0]])

# Calculate IoU
iou_scores = iou_width_height(boxes1, boxes2)

# Print the IoU scores
print("IoU Scores:")
print(iou_scores)





'''

The subtraction of the intersection area from the sum of the areas of the two bounding boxes is part of the formula for calculating the union of the bounding boxes' areas in the Intersection over Union (IoU) calculation.

Here's why the subtraction is necessary:

The union of two bounding boxes is defined as the total area covered by both bounding boxes. When you calculate the union, you add the areas of both boxes but then subtract the intersection area to avoid double-counting it.

The formula for the union is as follows:

Union Area = (Area of Box 1) + (Area of Box 2) - (Intersection Area)

(Area of Box 1) is represented by boxes1[..., 0] * boxes1[..., 1], which is the width multiplied by the height of the first bounding box.

(Area of Box 2) is represented by boxes2[..., 0] * boxes2[..., 1], which is the width multiplied by the height of the second bounding box.

(Intersection Area) is the area of overlap between the two bounding boxes, which is computed earlier as intersection.

By subtracting the intersection area, you ensure that the overlapping region is counted only once in the union calculation. This is important because if you don't subtract the intersection area, it would be counted twice, leading to an overestimation of the total area.

So, the subtraction of the intersection area is a crucial step in accurately calculating the union of bounding boxes, which is used to compute the IoU (Intersection over Union) score, a common metric in object detection and computer vision tasks.'''
















