from functions import *

@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Среднеквадратичная ошибка
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Ошибка структурного сходства SSIM
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    
    # Штрафы за максимумы и минимумы
    max_penalty = tf.square(tf.reduce_max(y_true) - tf.reduce_max(y_pred))
    min_penalty = tf.square(tf.reduce_min(y_true) - tf.reduce_min(y_pred))
    
    # Лапласиан для оценки локальных максимумов и минимумов
    laplacian_true = tf.image.sobel_edges(y_true)
    laplacian_pred = tf.image.sobel_edges(y_pred)
    laplacian_loss = tf.reduce_mean(tf.square(laplacian_true - laplacian_pred))
    
    # Расчет гистограмм
    hist_true = tf.histogram_fixed_width(y_true, [0, 1], nbins=10)
    hist_pred = tf.histogram_fixed_width(y_pred, [0, 1], nbins=10)
    
    # Приведение типов гистограмм к float32
    hist_true = tf.cast(hist_true, tf.float32)
    hist_pred = tf.cast(hist_pred, tf.float32)
    
    # Earth Mover's Distance для сравнения гистограмм
    emd_loss = tf.reduce_mean(tf.abs(tf.cumsum(hist_true) - tf.cumsum(hist_pred)))

    # Комбинированная функция потерь
    return mse + ssim_loss + 0.5 * (max_penalty + min_penalty) + 0.5 * laplacian_loss + 0.5 * emd_loss


def residual_block(x, filters, kernel_size=3):
    """Блок с остаточной связью."""
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Слой внимания
    attention = Conv2D(filters, kernel_size, activation='sigmoid', padding='same')(x)
    x = x * attention
    
    x = add([shortcut, x])  # Остаточная связь
    x = Activation('relu')(x)
    return x

def autoencoder():
    """Создание архитектуры сети."""
    input_img = Input(shape=(None, None, 1))

    # Энкодер
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = residual_block(x, 64)
        
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = residual_block(x, 128)

    # Декодер
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = residual_block(x, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = residual_block(x, 64)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    optimizer = tf.optimizers.Adam(learning_rate=0.001, weight_decay=0.001)
    autoencoder.compile(optimizer=optimizer, loss=custom_loss)

    autoencoder.summary()

    # Путь для сохранения логов TensorBoard
    log_dir = "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    return autoencoder, tensorboard_callback
