
import os
import cv2
import numpy as np
from tqdm import tqdm
from config import IMG_SIZE, MAX_BOXES_PER_IMAGE, MIN_FACE_SIZE, OUTPUT_DIR, WIDER_ROOT


def parse_annotations(annotation_file, images_root):
    records = []
    with open(annotation_file, "r") as f:
        lines = [l.strip() for l in f]

    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        rel_path = lines[i]
        i += 1
        if i >= len(lines):
            break
        try:
            n_faces = int(lines[i])
            i += 1
        except ValueError:
            i += 1
            continue

        boxes = []
        for _ in range(max(n_faces, 1)):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            if w > 0 and h > 0:
                boxes.append([x1, y1, w, h])

        records.append(
            (os.path.join(images_root, rel_path), np.array(boxes, dtype=np.float32).reshape(-1, 4))
        )
    return records


def _to_normalized_cxcywh(boxes_xywh, orig_w, orig_h, img_size):
    sx, sy = img_size / orig_w, img_size / orig_h
    out = np.zeros_like(boxes_xywh)
    out[:, 0] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * sx / img_size
    out[:, 1] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * sy / img_size
    out[:, 2] = boxes_xywh[:, 2] * sx / img_size
    out[:, 3] = boxes_xywh[:, 3] * sy / img_size
    return np.clip(out, 0.0, 1.0)


def build_detection_split(
    annotation_file,
    images_root,
    output_path,
    img_size=IMG_SIZE,
    min_face_size=MIN_FACE_SIZE,
    max_boxes=MAX_BOXES_PER_IMAGE,
):
    records = parse_annotations(annotation_file, images_root)

    images = []
    all_boxes = []

    for img_path, boxes in tqdm(records, desc=os.path.basename(output_path)):
        if not os.path.isfile(img_path):
            continue
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        valid = boxes[(boxes[:, 2] >= min_face_size) & (boxes[:, 3] >= min_face_size)]
        if len(valid) == 0:
            continue  

        if len(valid) > max_boxes:
            areas = valid[:, 2] * valid[:, 3]
            keep = np.argsort(-areas)[:max_boxes]
            valid = valid[keep]

        resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        cxcywh = _to_normalized_cxcywh(valid, w, h, img_size).astype(np.float32)

        images.append(resized)
        all_boxes.append(cxcywh)

    images_arr = np.array(images, dtype=np.uint8)
    boxes_obj = np.empty(len(all_boxes), dtype=object)
    for idx, b in enumerate(all_boxes):
        boxes_obj[idx] = b

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, images=images_arr, boxes=boxes_obj)


def preprocess_and_save():
    for split, ann, imgs in [
        (
            "train",
            os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_train_bbx_gt.txt"),
            os.path.join(WIDER_ROOT, "WIDER_train", "images"),
        ),
        (
            "val",
            os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_val_bbx_gt.txt"),
            os.path.join(WIDER_ROOT, "WIDER_val", "images"),
        ),
    ]:
        build_detection_split(ann, imgs, os.path.join(OUTPUT_DIR, f"{split}.npz"))


if __name__ == "__main__":
    preprocess_and_save()
