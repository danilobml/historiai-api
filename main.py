from flask import Flask, request, jsonify
# from llms.llama2_70 import send_prompt_to_llama2_70
from ocr.tesseract import extract_text_from_image, allowed_file
from llms.openai import generate_summary


app = Flask(__name__)


@app.route('/')
def home():
    return """
        Welcome to the HistoriAI (demo) API!

        Available endpoints:

        - /summarize : receives an image file, runs ocr and returns
        a summarization from the LLM.
    """


@app.route('/summarize', methods=['POST'])
def post_image_to_ocr():
    file = request.files['file']

    if file and allowed_file(file.filename):
        text = extract_text_from_image(file)
        result = generate_summary(text)
        return jsonify(result), 200
    elif not file:
        return jsonify({"Error": "No file uploaded."}), 400
    else:
        return jsonify({"Error": "File type not allowed."}), 400


if __name__ == "__main__":
    app.run(debug=True)
