from transformers import Yolov5ForImageRecognition, ViTFeatureExtractor
import pytesseract
from visualize import visualize_prediction

def load_models():
    model = Yolov5ForImageRecognition.from_pretrained("ultralytics/yolov5")
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/detector-vit-fpn")
    return model, feature_extractor

def preprocess_image(img, feature_extractor):
    inputs = feature_extractor(img, return_tensors="pt")
    return inputs

def detect_license_plate(img, feature_extractor, model):
    inputs = preprocess_image(img, feature_extractor)
    output_dict = model(inputs)
    return output_dict

def recognize_characters(img):
    img = img.convert('L')
    text = pytesseract.image_to_string(img)
    return text

def process_image(img, feature_extractor, model):
    output_dict = detect_license_plate(img, feature_extractor, model)
    result_img = visualize_prediction(img, output_dict, threshold=0.5, id2label={1: "license-plate"})

    for i, box in enumerate(output_dict["boxes"]):
        if output_dict["labels"][i] == 1: # 1 corresponds to the license plate label
            xmin, ymin, xmax, ymax = box
            plate_img = img.crop((xmin, ymin, xmax, ymax))
            plate_text = recognize_characters(plate_img)
            print("License plate text:", plate_text)

    return result_img
