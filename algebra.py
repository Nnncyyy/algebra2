import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(ruta):
    """Carga una imagen desde una ruta especificada."""
    imagen = cv2.imread(ruta)
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return None
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def mostrar_imagenes(original, transformada, titulo_transformacion):
    """Muestra la imagen original y transformada lado a lado."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformada)
    plt.title(f"Transformación: {titulo_transformacion}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def rotacion(imagen, angulo):
    """Aplica una rotación a la imagen."""
    filas, columnas, _ = imagen.shape
    matriz_rotacion = cv2.getRotationMatrix2D((columnas // 2, filas // 2), angulo, 1)
    return cv2.warpAffine(imagen, matriz_rotacion, (columnas, filas))

def escalado(imagen, factor):
    """Aplica un escalado a la imagen."""
    return cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

def reflexion(imagen, eje):
    """Aplica una reflexión a la imagen."""
    if eje == 'horizontal':
        return cv2.flip(imagen, 1)
    elif eje == 'vertical':
        return cv2.flip(imagen, 0)
    else:
        print("Error: Eje no válido. Usa 'horizontal' o 'vertical'.")
        return imagen

def traslacion(imagen, desplazamiento_x, desplazamiento_y):
    """Aplica una traslación a la imagen."""
    filas, columnas, _ = imagen.shape
    matriz_traslacion = np.float32([[1, 0, desplazamiento_x], [0, 1, desplazamiento_y]])
    return cv2.warpAffine(imagen, matriz_traslacion, (columnas, filas))

def guardar_imagen(ruta, imagen):
    """Guarda la imagen transformada en un archivo."""
    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    cv2.imwrite(ruta, imagen_bgr)
    print(f"Imagen guardada en {ruta}")

def menu():
    """Menú interactivo para seleccionar la transformación."""
    ruta = input("Introduce la ruta de la imagen: ")
    imagen_original = cargar_imagen(ruta)
    if imagen_original is None:
        return

    while True:
        print("\n--- Menú de Transformaciones ---")
        print("1. Rotación")
        print("2. Escalado")
        print("3. Reflexión")
        print("4. Traslación")
        print("5. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            angulo = float(input("Introduce el ángulo de rotación (en grados): "))
            imagen_transformada = rotacion(imagen_original, angulo)
            mostrar_imagenes(imagen_original, imagen_transformada, f"Rotación {angulo}°")
        elif opcion == "2":
            factor = float(input("Introduce el factor de escalado (e.g., 0.5 para reducir, 2 para ampliar): "))
            imagen_transformada = escalado(imagen_original, factor)
            mostrar_imagenes(imagen_original, imagen_transformada, f"Escalado x{factor}")
        elif opcion == "3":
            eje = input("Introduce el eje de reflexión ('horizontal' o 'vertical'): ").lower()
            imagen_transformada = reflexion(imagen_original, eje)
            mostrar_imagenes(imagen_original, imagen_transformada, f"Reflexión {eje}")
        elif opcion == "4":
            desplazamiento_x = int(input("Introduce el desplazamiento en X (en píxeles): "))
            desplazamiento_y = int(input("Introduce el desplazamiento en Y (en píxeles): "))
            imagen_transformada = traslacion(imagen_original, desplazamiento_x, desplazamiento_y)
            mostrar_imagenes(imagen_original, imagen_transformada, f"Traslación ({desplazamiento_x}, {desplazamiento_y})")
        elif opcion == "5":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
        
        guardar = input("¿Quieres guardar la imagen transformada? (s/n): ").lower()
        if guardar == "s":
            ruta_guardado = input("Introduce la ruta para guardar la imagen (e.g., transformada.jpg): ")
            guardar_imagen(ruta_guardado, imagen_transformada)

if __name__ == "__main__":
    menu()