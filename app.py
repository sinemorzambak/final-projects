import gradio as gr
from PIL import Image
from license_plate_recognition import load_models, process_image

model, feature_extractor = load_models()

def recognize_license_plate(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    result_img = process_image(img, feature_extractor, model)
    return result_img

iface = gr.Interface(fn=recognize_license_plate, inputs="image", outputs="image")
iface.launch()
