from functions import *
from core import *

high_res_folder = "/data/200_512_cleared/"
low_res_folder = "/data/1000_512_cleared/" 
modelpath = "/models/"
modelname = "AE_200_1000.keras"

# Создание экземпляра Callback класса
class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, input_images, output_images, image_id, test_im1, test_im2):
        self.input_images = input_images
        self.output_images = output_images
        self.image_id = image_id  
        self.test_im1 = test_im1
        self.test_im2 = test_im2        
    
    def on_epoch_end(self, epoch, logs=None):
        # Получение конкретного изображения по ID
        input_image = self.input_images[self.image_id]
        output_image = self.output_images[self.image_id]
        
        # Применение predict к изображению
        p_im = autoencoder_model.predict(np.expand_dims(input_image, axis=0))
        
        # Определение глобальных минимальных и максимальных значений для цветовой шкалы
        vmin = min(input_image.min(), output_image.min())
        vmax = max(input_image.max(), output_image.max())
        
        # Визуализация результатов с единой шкалой
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im0 = axes[0].imshow(input_image[:, :, 0], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[0].set_title('High Resolution')
        fig.colorbar(im0, ax=axes[0], orientation='vertical')

        im1 = axes[1].imshow(output_image[:, :, 0], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth')
        fig.colorbar(im1, ax=axes[1], orientation='vertical')

        im2 = axes[2].imshow(p_im[0, :, :, 0], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Generated')
        fig.colorbar(im2, ax=axes[2], orientation='vertical')

        fig.tight_layout()
        plt.savefig("/kaggle/working/vis/vis_epoch_"+str(epoch+1)+".png")
        plt.close()
        
        pred_test_image1 = autoencoder_model.predict(self.test_im1)
        im1 = Image.fromarray(pred_test_image1[0, :, :, 0], mode='F')
        im1.save('/kaggle/working/autorun_test/avtoencoder_'+str(epoch+1)+'.tif')
        
        pred_test_image2 = autoencoder_model.predict(self.test_im2)
        im2 = Image.fromarray(pred_test_image2[0, :, :, 0], mode='F')
        im2.save('/kaggle/working/Samsonov_DEM_test/avtoencoder_'+str(epoch+1)+'.tif')
        
        autoencoder_model.save(modelpath + 'Check_e'+str(epoch) + modelname)
        
    
    if __name__ == "__main__":
        autoencoder_model, tensorboard_callback = autoencoder()
        # autoencoder_model = load_model(modelpath + modelname) #If it already exists

        # high_res = preparing_data(high_res_folder) 
        # low_res = preparing_data(low_res_folder) 

        high_res = prepare_and_augment_data(high_res_folder)  
        low_res = prepare_and_augment_data(low_res_folder) 
        test1_image = preparing_largedata('/kaggle/input/dem-test/data_full_test')
        test2_image = preparing_largedata('/kaggle/input/dem-test/test')


        visualization_callback = VisualizationCallback(high_res, low_res, image_id=32, test_im1=test1_image, test_im2= test2_image)
        checkpoint_callback = callbacks.ModelCheckpoint(modelpath + modelname)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


        # Обучение модели
        autoencoder_model.fit(high_res, low_res, epochs=1000, batch_size=8, shuffle=True, validation_split=0.1, callbacks=[reduce_lr, checkpoint_callback, visualization_callback, tensorboard_callback])
        autoencoder_model.save(modelpath + modelname)

        print ("Model saved.")