from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import os
from PIL import Image

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
    image = np.array(Image.open(filepath).convert("RGB"))

    output_filename = 'output_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    if request.method == 'POST':
        transform_type = request.form['transformation']
        if transform_type == 'rotate':
            angle = float(request.form['angle'])
            image = manual_rotate_image(image, angle)
        elif transform_type == 'scale':
            factor = float(request.form['factor'])
            image = manual_scale_image(image, factor)
        elif transform_type == 'reflect':
            axis = request.form['axis']
            image = manual_reflect_image(image, axis)
        elif transform_type == 'translate':
            dx = int(request.form['dx'])
            dy = int(request.form['dy'])
            image = manual_translate_image(image, dx, dy)

        output_image = Image.fromarray(image.astype('uint8'))
        output_image.save(output_path)

    return render_template('transform.html', original_filename=filename, transformed_filename=output_filename)

@app.route('/static/images/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def manual_rotate_image(image, angle):
    angle_rad = np.deg2rad(angle)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)

    rows, cols, _ = image.shape

    #nuevas dimensiones
    diagonal = int(np.sqrt(rows**2 + cols**2))
    new_rows = new_cols = diagonal
    expanded_image = np.ones((new_rows, new_cols, 3), dtype=image.dtype) * 255  #blanco

    # Centrar imaen
    center_x, center_y = new_cols // 2, new_rows // 2
    offset_x, offset_y = (new_cols - cols) // 2, (new_rows - rows) // 2
    expanded_image[offset_y:offset_y + rows, offset_x:offset_x + cols] = image

    rotated_image = np.ones_like(expanded_image) * 255 

    # Aplicar la matriz de rotación
    for y in range(new_rows):
        for x in range(new_cols):
            x_shifted = x - center_x
            y_shifted = y - center_y

            orig_x = int(cos_val * x_shifted - sin_val * y_shifted + cols // 2)
            orig_y = int(sin_val * x_shifted + cos_val * y_shifted + rows // 2)

            if 0 <= orig_x < cols and 0 <= orig_y < rows:
                rotated_image[y, x] = expanded_image[orig_y + offset_y, orig_x + offset_x]

    return rotated_image


def manual_scale_image(image, factor):
    rows, cols, _ = image.shape
    new_rows, new_cols = int(rows * factor), int(cols * factor)

    new_image = np.zeros((new_rows, new_cols, 3), dtype=image.dtype)

    for y in range(new_rows):
        for x in range(new_cols):
            orig_x = int(x / factor)
            orig_y = int(y / factor)
            if orig_x < cols and orig_y < rows:
                new_image[y, x] = image[orig_y, orig_x]

    return new_image

def manual_reflect_image(image, axis):
    rows, cols, _ = image.shape
    new_image = np.zeros_like(image)

    if axis == 'horizontal':
        for y in range(rows):
            for x in range(cols):
                new_image[y, cols - 1 - x] = image[y, x]
    elif axis == 'vertical':
        for y in range(rows):
            for x in range(cols):
                new_image[rows - 1 - y, x] = image[y, x]

    return new_image

def manual_translate_image(image, dx, dy):
    rows, cols, _ = image.shape

    #estetica
    expanded_rows = rows + abs(dy)
    expanded_cols = cols + abs(dx)
    expanded_image = np.ones((expanded_rows, expanded_cols, 3), dtype=image.dtype) * 255  # Fondo blanco

    offset_y = abs(dy) if dy < 0 else 0
    offset_x = abs(dx) if dx < 0 else 0
    expanded_image[offset_y:offset_y + rows, offset_x:offset_x + cols] = image

    #nueva imagen trasladada
    translated_image = np.ones_like(expanded_image) * 255  # Fondo blanco

    for y in range(expanded_rows):
        for x in range(expanded_cols):
            orig_x = x - dx
            orig_y = y - dy

            if 0 <= orig_x < cols + abs(dx) and 0 <= orig_y < rows + abs(dy):
                translated_image[y, x] = expanded_image[orig_y, orig_x]

    return translated_image

if __name__ == '__main__':
    app.run(debug=True, port=5002)
