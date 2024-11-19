import gradio as gr
from PIL import Image
import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
    ViltProcessor,
    ViltForQuestionAnswering
)
import warnings
warnings.filterwarnings('ignore')

# Load models
print("Loading models...")

# Image classification models
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
image_classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Image captioning models
caption_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
caption_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# VQA models
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

print("Models loaded successfully.")

def analyze_image(image, question):
    try:
        # Ensure image is in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Image Classification
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = image_classifier(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        classifications = {
            image_classifier.config.id2label[top5_catid[i].item()]: f"{top5_prob[i].item():.4f}"
            for i in range(top5_prob.size(0))
        }

        # Image Captioning
        inputs = caption_processor(images=image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Visual Question Answering
        if question.strip() == "":
            answer = "No question provided."
        else:
            encoding = vqa_processor(image, question, return_tensors="pt")
            outputs = vqa_model(**encoding)
            logits = outputs.logits
            answer_idx = logits.argmax(-1).item()
            answer = vqa_model.config.id2label[answer_idx].capitalize()

        return classifications, caption, answer

    except Exception as e:
        return {"Error": str(e)}, "An error occurred during processing.", "An error occurred during processing."

# Enhanced CSS for better appearance
css = """
body {
    background-color: #f7f9fc !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    color: #333 !important;
}
h1 {
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    color: #2c3e50 !important;
    margin-bottom: 10px !important;
}
p.description {
    text-align: center !important;
    font-size: 1.2rem !important;
    color: #34495e !important;
    margin-bottom: 30px !important;
}
.gradio-container {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    padding: 40px !important;
    max-width: 900px !important;
    margin: 40px auto !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
}
.gr-button {
    background-color: #3498db !important;
    color: #fff !important;
    border: none !important;
    padding: 14px 24px !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
}
.gr-button:hover {
    background-color: #2980b9 !important;
}
.gr-input, .gr-output {
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
}
.gr-image-preview {
    border-radius: 8px !important;
}
.gr-label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #2c3e50 !important;
    margin-bottom: 10px !important;
}
.gr-textbox textarea {
    font-size: 1rem !important;
    line-height: 1.5 !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1>Image Analyzer with Classification, Captioning, and VQA</h1>")
    gr.Markdown("<p class='description'>Upload an image to receive classifications, a caption, and ask a question about it.</p>")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    with gr.Row():
        question_input = gr.Textbox(label="Ask a question about the image (optional)", placeholder="What is the color of the car?")

    with gr.Row():
        analyze_button = gr.Button("Analyze Image")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Label("Top Classifications")
            classification_output = gr.Label()
            gr.Label("Generated Caption")
            caption_output = gr.Textbox(lines=2, max_lines=2, label="", interactive=False)
        with gr.Column(scale=1):
            gr.Label("Answer to Your Question")
            answer_output = gr.Textbox(lines=2, max_lines=2, label="", interactive=False)

    analyze_button.click(
        analyze_image,
        inputs=[image_input, question_input],
        outputs=[classification_output, caption_output, answer_output]
    )

if __name__ == "__main__":
    demo.launch()
