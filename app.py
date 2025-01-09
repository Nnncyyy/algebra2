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
    image = np.array(Image.open(filepath).convert("RGB"))  # Cargar la imagen con PIL y convertir a RGB

    output_filename = 'output_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    if request.method == 'POST':
        transform_type = request.form['transformation']
        if transform_type == 'rotate':
            angle = float(request.form['angle'])  # Obtener el ángulo desde el formulario
            image = rotate_image_with_background(image, angle)  # Rota la imagen con fondo visible

        # Guardar la imagen resultante
        output_image = Image.fromarray(image.astype('uint8'))
        output_image.save(output_path)

    return render_template('transform.html', original_filename=filename, transformed_filename=output_filename)


@app.route('/static/images/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def rotation_matrix(angle):
    """Crea una matriz de rotación."""
    angle_rad = np.deg2rad(angle)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0],
        [0, 0, 1]
    ])

def bicubic_interpolation(image, x, y):
    """Interpolación bicúbica para obtener el valor de los píxeles."""
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))

    # Limitar las coordenadas para no acceder fuera de los límites
    x2 = min(x1 + 1, image.shape[1] - 1)
    y2 = min(y1 + 1, image.shape[0] - 1)
    
    dx = x - x1
    dy = y - y1

    def cubic(t):
        return (1.5 * abs(t) ** 3 - 2.5 * abs(t) ** 2 + 1) if abs(t) < 1 else (-0.5 * abs(t) ** 3 + 2.5 * abs(t) ** 2 - 4 * abs(t) + 2) if abs(t) < 2 else 0

    # Bicubic interpolation on each channel (R, G, B)
    top_left = image[y1, x1]
    top_right = image[y1, x2]
    bottom_left = image[y2, x1]
    bottom_right = image[y2, x2]
    
    top = top_left + dx * (top_right - top_left)
    bottom = bottom_left + dx * (bottom_right - bottom_left)

    return top + dy * (bottom - top)

def rotate_image_with_background(image, angle):
    """Rota la imagen con fondo blanco para que toda la imagen se visualice, usando interpolación bicúbica."""
    rows, cols, _ = image.shape

    # Convertir el ángulo de grados a radianes
    angle_rad = np.deg2rad(angle)

    # Calcular las nuevas dimensiones para evitar el recorte
    diagonal = int(np.sqrt(rows**2 + cols**2))  # La diagonal es el tamaño máximo de la nueva imagen
    padding_vertical = (diagonal - rows) // 2
    padding_horizontal = (diagonal - cols) // 2

    # Crear una nueva imagen expandida (fondo blanco)
    new_rows = diagonal
    new_cols = diagonal
    expanded_image = np.ones((new_rows, new_cols, 3), dtype=np.uint8) * 255  # Fondo blanco (valor 255)

    # Ajustar la imagen original en el centro del fondo expandido
    start_row = padding_vertical
    start_col = padding_horizontal
    expanded_image[start_row:start_row+rows, start_col:start_col+cols] = image

    # Matriz de rotación
    rotation_matrix_2d = rotation_matrix(angle)

    # Crear la nueva imagen después de la rotación
    rotated_image = np.zeros_like(expanded_image)

    # Calcular el centro de la nueva imagen
    center_x, center_y = new_cols // 2, new_rows // 2

    # Aplicar la matriz de rotación a cada píxel de la imagen expandida
    for y in range(new_rows):
        for x in range(new_cols):
            # Coordenadas del píxel
            coord = np.array([x - center_x, y - center_y, 1])
            new_coord = np.dot(rotation_matrix_2d, coord)

            # Nuevas coordenadas después de rotación
            new_x = int(new_coord[0] + center_x)
            new_y = int(new_coord[1] + center_y)

            # Asegurarse de que las coordenadas estén dentro de los límites
            if 0 <= new_x < new_cols and 0 <= new_y < new_rows:
                # Aplicar interpolación bicúbica
                rotated_image[y, x] = bicubic_interpolation(expanded_image, new_x, new_y)

    return rotated_image

def scaling_matrix(factor):
    """Crea una matriz de escalado."""
    return np.array([
        [factor, 0, 0],
        [0, factor, 0],
        [0, 0, 1]
    ])

def reflection_matrix(axis):
    """Crea una matriz de reflexión."""
    if axis == 'horizontal':
        return np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif axis == 'vertical':
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    return np.eye(3)

def translation_matrix(dx, dy):
    """Crea una matriz de traslación."""
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def apply_manual_matrix(image, matrix):
    """Aplica una transformación utilizando una matriz."""
    rows, cols, _ = image.shape
    center_x, center_y = cols // 2, rows // 2

    # Crear nueva imagen
    new_image = np.zeros_like(image)

    # Matriz inversa para recorrer los píxeles
    inv_matrix = np.linalg.inv(matrix)

    for y in range(rows):
        for x in range(cols):
            # Transformar las coordenadas
            coord = np.array([x - center_x, y - center_y, 1])
            new_coord = np.dot(inv_matrix, coord)
            new_x = int(new_coord[0] + center_x)
            new_y = int(new_coord[1] + center_y)

            # Copiar el píxel si está dentro de los límites
            if 0 <= new_x < cols and 0 <= new_y < rows:
                new_image[y, x] = image[new_y, new_x]

    return new_image

import cv2

def manual_scale_image(image, factor):
    """Escala la imagen usando OpenCV."""
    # Redimensionar la imagen con el factor de escala.
    rows, cols, _ = image.shape
    new_rows = int(rows * factor)
    new_cols = int(cols * factor)

    # Redimensionar la imagen.
    scaled_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    return scaled_image


if __name__ == '__main__':
    app.run(debug=True, port=5002)

