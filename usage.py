# from functions import *
import os
import rasterio
from rasterio.enums import Resampling
from tensorflow.keras.models import load_model
import numpy as np
from core import *


def is_tif_file(path):
    return path.endswith('.tif')

def listdir_nohidden(path):
    return sorted([f for f in os.listdir(path) if not f.startswith('.') and is_tif_file(f)])

def process_and_save_images(input_path, model_path, model_name, output_file):
    # Определение изображений для обработки
    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in listdir_nohidden(input_path)]
    elif os.path.isfile(input_path) and is_tif_file(input_path):
        image_paths = [input_path]
    else:
        raise ValueError("Provided path is neither a directory nor a .tif file.")
    
    print("Count of images to process: " + str(len(image_paths)))

    # Подготовка данных изображений
    images = []
    metadata_list = []
    for file_path in image_paths:
        with rasterio.open(file_path) as src:
            img_height, img_width = src.shape
            img_channels = 1 
            img = src.read(
                out_shape=(img_channels, img_height, img_width),
                resampling=Resampling.bilinear
            )
            img_reshaped = np.expand_dims(img, axis=-1)
            images.append(img_reshaped)
            metadata_list.append(src.meta)

    # Конвертация в NumPy массив и нормализация
    images_np = np.concatenate(images, axis=0).astype('float32') / 10000.0
    print(images_np.shape)

    # Загрузка и применение модели
    model = load_model(os.path.join(model_path, model_name))
    predictions = model.predict(images_np)

    # Сохранение предсказанных изображений с геопривязкой
    for i, prediction in enumerate(predictions):
        prediction_rescaled = prediction[:, :, 0] * 10000.0  # Обратная нормализация
        meta = metadata_list[i]
        meta.update(dtype=rasterio.float32, count=1)

        output_path = os.path.join(output_file+f'pred_{i}.tif')
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(prediction_rescaled, 1)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    process_and_save_images(
        input_path='test/Moskva_0200.tif',
        model_path='trained_models/',
        model_name='AE_200_500.keras',
        output_file='test/res/test.tif'
        )