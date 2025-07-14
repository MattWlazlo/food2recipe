from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

print(generate_caption("/home/mwlazlo/food2recipe/sample-images/apples.jpg"))
print(generate_caption("/home/mwlazlo/food2recipe/sample-images/fruits.jpg"))
print(generate_caption("/home/mwlazlo/food2recipe/sample-images/fruits2.jpg"))
print(generate_caption("/home/mwlazlo/food2recipe/sample-images/grapes.jpg"))
print(generate_caption("/home/mwlazlo/food2recipe/sample-images/kiwi.jpg"))
print(generate_caption("/home/mwlazlo/food2recipe/sample-images/labuffy.png"))


