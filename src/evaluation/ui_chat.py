#!/usr/bin/env python3
import os
import sys
import torch
from flask import Flask, request, redirect, url_for, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

# Import your MiniLLaVA model
from src.models.vlm import MiniLLaVA

# Global variables
MODEL = None
TOKENIZER = None
PREFIX_EMBEDS = None
CURRENT_IMAGE = None  # holds the filename (relative URL) of the uploaded image

# Directory for uploads (placed in src/evaluation/uploads)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_model():
    global MODEL, TOKENIZER, PREFIX_EMBEDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading MiniLLaVA model...")
    MODEL = MiniLLaVA(device=device)
    MODEL.to(device)
    
    # Inject LoRA adapters as in training.
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    MODEL.llm.model = get_peft_model(MODEL.llm.model, lora_config)
    
    # Load the trained LLM checkpoint.
    checkpoint_path = os.path.join(BASE_DIR, "../../saved_models/checkpoint_epoch10.pth")
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load LLM weights
        if "llm_state_dict" in checkpoint:
            MODEL.llm.load_state_dict(checkpoint["llm_state_dict"])
        else:
            print("[WARNING] 'llm_state_dict' not found in the checkpoint.")
        
        # Load projector weights
        if "projector_state_dict" in checkpoint:
            MODEL.projection.load_state_dict(checkpoint["projector_state_dict"])
        else:
            print("[WARNING] 'projector_state_dict' not found in the checkpoint.")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
    
    # Load pretrained projector checkpoint.
    pretrained_projector_path = os.path.join(BASE_DIR, "../../saved_models/projector_iter_102300_epoch31.pth")
    if os.path.exists(pretrained_projector_path):
        print(f"[INFO] Loading pretrained projector from {pretrained_projector_path}")
        MODEL.projection.load_state_dict(torch.load(pretrained_projector_path, map_location=device))
    else:
        print("[WARNING] Pretrained projector not found. Using default projector weights.")
    
    TOKENIZER = MODEL.llm.tokenizer
    if TOKENIZER.pad_token is None:
        print("[INFO] Setting tokenizer.pad_token to tokenizer.eos_token")
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL.eval()
    PREFIX_EMBEDS = None

# Load model on startup.
load_model()

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# HTML template (using Bootstrap) for a two-column chat UI.
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MiniLLaVA Chat Interface</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body { background-color: #f8f9fa; }
    .card { margin-bottom: 20px; }
    #chat_log { height: 400px; overflow-y: scroll; background: white; padding: 10px; border: 1px solid #dee2e6; }
    .chat-msg { margin-bottom: 10px; }
    .chat-msg .user { font-weight: bold; }
    .chat-msg .assistant { color: #007bff; }
  </style>
</head>
<body>
<div class="container mt-4">
  <h2 class="mb-4 text-center">MiniLLaVA Chat Interface</h2>
  <div class="row">
    <!-- Left Column: Image display -->
    <div class="col-md-4">
      <div class="card">
        <div class="card-header bg-info text-white">Uploaded Image</div>
        <div class="card-body text-center">
          {% if current_image %}
            <img src="{{ url_for('uploaded_file', filename=current_image) }}" class="img-fluid" alt="Uploaded Image">
          {% else %}
            <p>No image uploaded.</p>
          {% endif %}
        </div>
        <div class="card-footer">
          <form action="/upload_image" method="post" enctype="multipart/form-data">
            <div class="custom-file">
              <input type="file" class="custom-file-input" id="image" name="image" accept="image/*" required>
              <label class="custom-file-label" for="image">Choose image</label>
            </div>
            <button type="submit" class="btn btn-primary btn-block mt-2">Upload Image</button>
          </form>
        </div>
      </div>
    </div>
    <!-- Right Column: Chat interface -->
    <div class="col-md-8">
      <div class="card">
        <div class="card-header bg-secondary text-white">Chat</div>
        <div class="card-body" id="chat_log">
          <!-- Chat messages will be appended here -->
        </div>
        <div class="card-footer">
          <form id="chat_form">
            <div class="input-group">
              <input type="text" class="form-control" id="prompt" placeholder="Enter your prompt" autocomplete="off" required>
              <div class="input-group-append">
                <button type="submit" class="btn btn-success">Send</button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</div>
<!-- Bootstrap and jQuery scripts -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script>
  // Update file input label when a file is selected
  $(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
  });
  
  // Handle chat submission
  document.getElementById("chat_form").onsubmit = async function(e) {
    e.preventDefault();
    const promptInput = document.getElementById("prompt");
    const prompt = promptInput.value;
    if (!prompt) return;
    const response = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: prompt})
    });
    const data = await response.json();
    const chatLog = document.getElementById("chat_log");
    chatLog.innerHTML += "<div class='chat-msg'><span class='user'>User:</span> " + prompt + "</div>";
    chatLog.innerHTML += "<div class='chat-msg'><span class='assistant'>Assistant:</span> " + data.response + "</div>";
    promptInput.value = "";
    chatLog.scrollTop = chatLog.scrollHeight;
  }
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    # Pass the current image filename (if any) to the template.
    return render_template_string(INDEX_HTML, current_image=CURRENT_IMAGE)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    global PREFIX_EMBEDS, CURRENT_IMAGE, MODEL
    if "image" not in request.files:
        return "No image part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    print(f"[INFO] Image saved to {filepath}")
    CURRENT_IMAGE = filename  # update global image filename
    # Process the image to compute visual prefix embeddings.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        image_emb = MODEL.vision_encoder.forward(filepath)
        PREFIX_EMBEDS = MODEL.projection(image_emb)
    return redirect(url_for("index"))

@app.route("/chat", methods=["POST"])
def chat():
    global PREFIX_EMBEDS, MODEL, TOKENIZER
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": ""})
    if PREFIX_EMBEDS is None:
        return jsonify({"response": "No image loaded. Please upload an image first."})
    # Format the prompt as in training.
    chat_prompt = f"user: {prompt}\nassistant:"
    print("chat prompt: ", chat_prompt)
    with torch.no_grad():
        generated_ids = MODEL.llm.generate(
            [chat_prompt],
            prefix_embeds=PREFIX_EMBEDS,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            pad_token_id=TOKENIZER.eos_token_id
        )
    generated_text = TOKENIZER.decode(generated_ids[0], skip_special_tokens=True)
    if generated_text.startswith(chat_prompt):
        generated_text = generated_text[len(chat_prompt):].strip()
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    # Run the app on localhost port 8888.
    app.run(host="0.0.0.0", port=8888)
