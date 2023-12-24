import pytesseract as pyt
import cv2
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# TODO - Check using tesseract api: https://pypi.org/project/tess-py-api/


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_image(image_file):
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # pyt.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

    pyt.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

    text = pyt.image_to_string(image)

    return text
