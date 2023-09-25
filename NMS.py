import torch

# Function to calculate Intersection over Union (IoU) between bounding boxes
def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    if box_format == "midpoint":
        # Calculate coordinates of the top-left (x1, y1) and bottom-right (x2, y2) corners for box1
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        # Calculate coordinates of the top-left (x1, y1) and bottom-right (x2, y2) corners for box2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        # Extract coordinates of the top-left (x1, y1) and bottom-right (x2, y2) corners for both boxes
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Calculate the coordinates of the intersection's top-left (x1, y1) and bottom-right (x2, y2) corners
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate the area of intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the areas of both bounding boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Calculate IoU (Intersection over Union) with epsilon added for numerical stability
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return iou

# Function for non-maximum suppression (NMS)
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list

    # Filter out boxes with confidence scores below the specified threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort bounding boxes by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # List to store the selected bounding boxes after NMS
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # Filter out overlapping boxes based on IoU threshold
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # Check if boxes belong to different classes
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),  # Coordinates of chosen_box
                torch.tensor(box[2:]),  # Coordinates of current box being compared
                box_format=box_format,
            )
               < iou_threshold
        ]

        # Append the chosen_box to the list of selected bounding boxes
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

# Example:
# Define a list of bounding boxes [class, confidence, x1, y1, x2, y2]
bboxes = [
    [0, 0.9, 10, 10, 30, 30],  # Box 1
    [0, 0.75, 15, 15, 35, 35],  # Box 2 (partially overlaps with Box 1)
    [1, 0.85, 5, 5, 25, 25],  # Box 3 (different class from Box 1)
]

# Define IoU threshold and confidence threshold
iou_threshold = 0.5
confidence_threshold = 0.8

# Apply non-maximum suppression
filtered_bboxes = non_max_suppression(bboxes, iou_threshold, confidence_threshold)

# Print the filtered bounding boxes after NMS
print("Filtered Bounding Boxes After NMS:")
print(filtered_bboxes)
