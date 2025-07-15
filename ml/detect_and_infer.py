from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
from PIL import Image

def detect_and_caption(image_path):
    # Open image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Load models
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    Blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    YOLO_model = YOLO("ml/yolo11n.pt")
    
    # Predicting captions with Blip2
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = Blip_model.generate(**inputs)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Predicting foods in image with YOLO11
    results = YOLO_model(image)
    for result in results:
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

    # Prints results to terminal and returns a tuple containing a list
    # of the objects detected, and the caption for the image
    print(names)    
    print(generated_text)
    return(names, generated_text)

# Used to make sure the function works, will remove later!
detect_and_caption("/home/mwlazlo/food2recipe/sample-images/apples.jpg")
    