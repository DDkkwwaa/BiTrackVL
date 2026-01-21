import os
import json
import numpy as np
from tqdm import tqdm

def load_bbox_file(path):
    """
    自动判断分隔符（逗号 / 空格 / tab）
    """
    with open(path, "r") as f:
        line = f.readline()

    # 自动检测分隔符
    if "," in line:
        delim = ","
    elif "\t" in line:
        delim = "\t"
    else:
        delim = None  # 空格

    try:
        data = np.loadtxt(path, delimiter=delim)
    except:
        data = np.genfromtxt(path, delimiter=delim)

    return data


def iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def calc_auc(iou_list):
    thresholds = np.linspace(0, 1, 101)
    success = [(iou_list >= t).mean() for t in thresholds]
    return float(np.mean(success))


def evaluate(attr_json, gt_root, pred_root):
    with open(attr_json, 'r') as f:
        attrs = json.load(f)

    attr_order = [
        "Illumination Variation",
        "Partial Occlusion",
        "Deformation",
        "Motion Blur",
        "Camera Motion",
        "Rotation",
        "Background Clutter",
        "Viewpoint Change",
        "Scale Variation",
        "Full Occlusion",
        "Fast Motion",
        "Out of View",
        "Low Resolution",
        "Aspect Ratio Change"
    ]

    results = {a: [] for a in attr_order}

    for seq, att_list in tqdm(attrs.items()):
        gt_file = os.path.join(gt_root, seq, "groundtruth.txt")
        pred_file = os.path.join(pred_root, seq + ".txt")

        if not (os.path.exists(gt_file) and os.path.exists(pred_file)):
            print("[WARN] 缺失文件:", seq)
            continue

        gt = load_bbox_file(gt_file)
        pred = load_bbox_file(pred_file)

        n = min(len(gt), len(pred))
        gt = gt[:n]
        pred = pred[:n]

        ious = np.array([iou(gt[i], pred[i]) for i in range(n)])
        auc = calc_auc(ious)

        for attr in att_list:
            results[attr].append(auc)

    return {a: round(float(np.mean(v)), 3) for a, v in results.items()}


if __name__ == "__main__":

    attrs_json = "lasot_test_attributes.json"

    gt_root = "/data/testing_dataset/LaSOT/"
    # pred_root = "/data/code_Lon/PycharmProjects/HIPB_up_large/output/test/tracking_results/hiptrack/hiptrack"
    # pred_root = "/data/code_Lon/PycharmProjects/HIPB_up_large/BiTrack_73.7_lasot"
    pred_root = "/data/code_Lon/PycharmProjects/HIPB_up_large/TAT"
    out = evaluate(attrs_json, gt_root, pred_root)

    print("\n===== LaSOT Attribute AUC =====")
    for k, v in out.items():
        print(f"{k:25s}: {v:.3f}")

    print("\n雷达图数组：")
    print([out[a] for a in out.keys()])
