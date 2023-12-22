from flask import Flask, request, jsonify
# from llms.llama2_70 import send_prompt_to_llama2_70
from ocr.tesseract import extract_text_from_image, allowed_file
from llms.openai_funcs import generate_summary, get_text_analysis


app = Flask(__name__)


@app.route('/')
def home():
    return """
        Welcome to the HistoriAI (demo) API! n\

        Available endpoints: n\

        - /summarize : receives an image file, runs ocr and returns
        a summarization from the LLM.
    """


@app.route('/summarize', methods=['POST'])
def post_image_to_summary():
    file = request.files['photo']

    if file and allowed_file(file.filename):
        text = extract_text_from_image(file)
        result = generate_summary(text)
        return jsonify(result), 200
    elif not file:
        return jsonify({"Error": "No file uploaded."}), 400
    else:
        return jsonify({"Error": "File type not allowed."}), 400


@app.route('/analysis', methods=['POST'])
def post_image_to_analysis():
    file = request.files['photo']
    question = request.form['question']

    if file and allowed_file(file.filename):
        text = extract_text_from_image(file)
        result = get_text_analysis(text_input=text, question=question)
        return jsonify(result), 200
    elif not file:
        return jsonify({"Error": "No file uploaded."}), 400
    else:
        return jsonify({"Error": "File type not allowed."}), 400


if __name__ == "__main__":
    app.run()
