from flask import Flask, render_template, request, url_for, send_from_directory
import cv2
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

def detect_face(image_path): # "sample.png"
    image = cv2.imread(image_path)
    model = cv2.CascadeClassifier("haarcascade_face.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(gray_image)
    # [3 2 10 2], [13 22 10 4],
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # detected_sample.png
    # Saving the processed/detected image to upload
    detected_image_path = os.path.join(app.config["UPLOAD_FOLDER"],"detected_" + os.path.basename(image_path))
    cv2.imwrite(detected_image_path, image)
    return detected_image_path

@app.route('/upload', methods= ["POST"])
def upload_file():
    file = request.files['file']
    # Saving original image to uploads
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    detected_image_path = detect_face(filepath)

    image_url = url_for('uploaded_file', filename=os.path.basename(detected_image_path))

    return render_template('index.html', image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False)


