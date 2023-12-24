from flask import Flask, request, jsonify, render_template
# from llms.llama2_controller import send_prompt_to_llama2_70
from ocr.tesseract import extract_text_from_image, allowed_file
from llms.openai_controller import generate_summary, get_text_analysis
import sys
__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('api_list.html')


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
