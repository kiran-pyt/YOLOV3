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
