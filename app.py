from flask import Flask, render_template, request, jsonify
import os
import whisper
import pickle
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import requests
import json
from werkzeug.utils import secure_filename
import threading
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'm4a', 'ogg', 'flac'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def summarize_chunk(chunk, model="llama3.2:3b"):
    prompt = f"Summarize the following meeting transcript:\n\n{chunk}"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except Exception as e:
        return f"Error: {str(e)}"

def extract_action_items(summaries):
    context = "\n\n".join([f"{s['chunk']}: {s['summary']}" for s in summaries])
    prompt = (
        "From the following meeting summaries, extract a list of clear, actionable tasks "
        "or to-do items discussed. Include who is responsible if mentioned.\n\n"
        f"{context}\n\n"
        "List the tasks in bullet points or numbered list."
    )
    return summarize_chunk(prompt)

def process_audio(task_id, filepath):
    try:
        processing_status[task_id]['status'] = 'transcribing'
        processing_status[task_id]['progress'] = 10
        
        # Load Whisper model and transcribe
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(filepath)
        full_text = result["text"]
        
        processing_status[task_id]['progress'] = 30
        processing_status[task_id]['status'] = 'filtering_sentences'
        
        # Split into sentences
        sentences = sent_tokenize(full_text)
        
        # Load embedding model
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embed_model.encode(sentences, batch_size=32, show_progress_bar=False)
        
        processing_status[task_id]['progress'] = 40
        
        # Load PCA and SVM models
        pca = joblib.load("pca_model.pkl")
        embeddings_pca = pca.transform(embeddings)
        
        with open("svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        
        # Predict labels
        predicted_labels = svm_model.predict(embeddings_pca)
        use_sent = [s for s, label in zip(sentences, predicted_labels) if label == 0]
        
        processing_status[task_id]['progress'] = 50
        processing_status[task_id]['status'] = 'generating_summaries'
        
        # Create chunks and summarize
        text = ' '.join(use_sent)
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
        
        summaries = []
        for idx, chunk in enumerate(chunks):
            summary = summarize_chunk(chunk)
            summaries.append({
                "chunk": f"Chunk {idx + 1}",
                "summary": summary
            })
            progress = 50 + (40 * (idx + 1) / len(chunks))
            processing_status[task_id]['progress'] = int(progress)
        
        processing_status[task_id]['progress'] = 90
        processing_status[task_id]['status'] = 'extracting_action_items'
        
        # Extract action items
        action_items = extract_action_items(summaries)
        
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['status'] = 'completed'
        processing_status[task_id]['summaries'] = summaries
        processing_status[task_id]['action_items'] = action_items
        
    except Exception as e:
        processing_status[task_id]['status'] = 'error'
        processing_status[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(filepath)
    
    # Initialize processing status
    processing_status[task_id] = {
        'status': 'queued',
        'progress': 0,
        'filename': filename
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_audio, args=(task_id, filepath))
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Invalid task ID'}), 404
    return jsonify(processing_status[task_id])

if __name__ == '__main__':
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    app.run(debug=True, host='0.0.0.0', port=5000)