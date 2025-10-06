from flask import Flask, request, jsonify
import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

max_token = 1024
model_path = "/data/zxy/EmbodiedBench/models/pixtral-12b"

# Load the custom model
class CustomModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_type = 'pixtral'

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load model and processor
        print(">> Loading model ...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(">> Model loaded.")

    def respond(self, prompt, image_path=None):
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Construct input
        chat = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image"}
            ]}
        ]

        prompt_text = self.processor.apply_chat_template(chat, add_generation_prompt=True)

        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        # Adjust dtypes
        for k, v in inputs.items():
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                inputs[k] = v.to(dtype=self.model.dtype, device=self.model.device)
            else:
                inputs[k] = v.to(device=self.model.device)

        # Generate output
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_token)

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

# Initialize Flask app and model
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = CustomModel(model_path=model_path)

@app.route('/process', methods=['POST'])
def process_request():
    if 'image' not in request.files or 'sentence' not in request.form:
        return jsonify({'error': 'Missing image or sentence'}), 400

    image = request.files['image']
    sentence = request.form['sentence']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Generate response from the model
    model_response = model.respond(sentence, image_path=image_path)

    return jsonify({'response': model_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23333)