import cv2
import numpy as np

def visualize_prediction(img, output_dict, threshold, id2label):
    img = np.array(img)
    for i, box in enumerate(output_dict["boxes"]):
        score = output_dict["scores"][i]
        label = output_dict["labels"][i]
        if score < threshold:
            continue
        color = tuple(np.random.randint(0, 256, 3, dtype=int))
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, thickness=2)
        cv2.putText(img, f"{id2label[label]}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return img
