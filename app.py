from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        original_filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('transform_image', filename=file.filename))

@app.route('/transform/<filename>', methods=['GET', 'POST'])
def transform_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_filename = 'output_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    if request.method == 'POST':
        transform_type = request.form['transformation']
        if transform_type == 'rotate':
            angle = float(request.form['angle'])
            image_rgb = rotate_image(image_rgb, angle)
        elif transform_type == 'scale':
            factor = float(request.form['factor'])
            image_rgb = scale_image(image_rgb, factor)
        elif transform_type == 'reflect':
            axis = request.form['axis']
            image_rgb = reflect_image(image_rgb, axis)
        elif transform_type == 'translate':
            dx = int(request.form['dx'])
            dy = int(request.form['dy'])
            image_rgb = translate_image(image_rgb, dx, dy)

        
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    
    return render_template('transform.html', original_filename=filename, transformed_filename=output_filename)


@app.route('/static/images/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def rotate_image(image, angle):
    rows, cols, _ = image.shape

    
    diagonal = int(np.sqrt(rows**2 + cols**2)) 
    padding_vertical = (diagonal - rows) // 2
    padding_horizontal = (diagonal - cols) // 2

    
    expanded_image = cv2.copyMakeBorder(
        image,
        padding_vertical,
        padding_vertical,
        padding_horizontal,
        padding_horizontal,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255] 
    )

    
    new_rows, new_cols, _ = expanded_image.shape
    matrix = cv2.getRotationMatrix2D((new_cols // 2, new_rows // 2), angle, 1)

    
    rotated_image = cv2.warpAffine(expanded_image, matrix, (new_cols, new_rows))

    return rotated_image
def scale_image(image, factor):
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

def reflect_image(image, axis):
    if axis == 'horizontal':
        return cv2.flip(image, 1)
    elif axis == 'vertical':
        return cv2.flip(image, 0)
    return image

def translate_image(image, dx, dy):
    rows, cols, _ = image.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (cols, rows))

if __name__ == '__main__':
    app.run(debug=True, port=5002)
