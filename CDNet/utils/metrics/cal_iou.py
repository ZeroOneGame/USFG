import numpy as np

def cal_b2b_iou(box1, box2):
    """
    :param box1:
    :param box2:
    :return:两个BBox间的IoU
    """
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    a1, b1, a2, b2 = box2[0], box2[1], box2[2], box2[3]
    # Overlap Top-Left
    x_inter1 = max(x1, a1)
    y_inter1 = max(y1, b1)
    # Overlap Right-Down
    x_inter2 = min(x2, a2)
    y_inter2 = min(y2, b2)
    # Overlap Width and Height
    width = x_inter2 - x_inter1
    height = y_inter2 - y_inter1

    # Calculate Overlap
    if width < 0 or height < 0:
        inter = 0
    else:
        inter = width * height

    # Calculate Union
    sbox1 = (x2 - x1) * (y2 - y1)
    sbox2 = (a2 - a1) * (b2 - b1)
    union = sbox1 + sbox2 - inter
    # Calculate IOU
    iou = inter / union
    return iou

if __name__ == "__main__":
    # box1 = [10.0, 15.0, 13.0, 20.0] # x1, y1, x2, y2
    # box2 = [11.0, 17.0, 15.0, 22.0] # a1, b1, a2, b2

    # box1 = [1, 1, 3, 3] # x1, y1, x2, y2
    # box2 = [5, 5, 7, 7] # a1, b1, a2, b2

    # box1 = [0, 0, 3, 3] # x1, y1, x2, y2
    # box2 = [2, 2, 3, 4] # a1, b1, a2, b2

    box1 = [1, 0, 3, 2] # x1, y1, x2, y2
    box2 = [0, 1, 2, 3] # a1, b1, a2, b2

    print(cal_b2b_iou(box1, box2))
    print(cal_b2b_iou(box2, box1))