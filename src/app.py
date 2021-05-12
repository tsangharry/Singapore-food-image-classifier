import logging
import re
import base64
import mistune
from io import BytesIO
from inference import load_model, resize_image, predict_image, return_class, return_probability
from flask import Flask, request, render_template, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
model = load_model('src/model.h5')
app.logger.info('model loaded')

with open('README.md', 'r') as f:
    readme_doc = f.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
            "model": "InceptionV3",
            "pretrained-on": "ImageNet",
            "input-size": "299x299x3",
            "num-classes": 12
        })

@app.route('/docs', methods=['GET'])
def readme():
    return render_template('docs.html', data=mistune.markdown(readme_doc))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        input_img = request.json
        image_data = re.sub("^data:image/.+;base64,", "", input_img)
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = resize_image(img)

        prediction_prob = predict_image(img, model)
        class_name = return_class(prediction_prob)
        proba = return_probability(prediction_prob) * 100
        proba = round(proba,2)
        app.logger.info('Predicted to be %s with %s percent probability', class_name, proba)

        return jsonify(food=class_name, probability=str(proba))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
