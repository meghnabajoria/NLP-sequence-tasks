from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the models from Hugging Face
translation_model = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    translated_text = translation_model(text)
    return jsonify(translated_text)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    summarized_text = summarization_model(text)
    print(summarized_text)
    return jsonify(summarized_text)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get("question")
    context = data.get("context")
    if not question or not context:
        return jsonify({"error": "No question or context provided"}), 400
    answer = qa_model(question=question, context=context)
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True)
