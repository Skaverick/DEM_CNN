from functions import *
from osgeo import gdal

#dataset shapes
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1


def split_raster(in_path, out_path):
    if in_path.endswith('.tif'):
        print("Single file")
        input_filename = os.path.basename(in_path)
        print (input_filename)
        output_filename = 'tile_'+ input_filename
        tile_size_x = IMG_WIDTH
        tile_size_y = IMG_HEIGHT
        ds = gdal.Open(in_path)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        
        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path)  + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                os.system(com_string)

    else:
        print ("Folder")
        n = listdir_nohidden(in_path)
        print (n)
        for i in n:
            input_filename = i
            print (input_filename)
            output_filename = 'tile_'+ input_filename
            tile_size_x = IMG_WIDTH
            tile_size_y = IMG_HEIGHT
            ds = gdal.Open(in_path + input_filename)
            band = ds.GetRasterBand(1)
            xsize = band.XSize
            ysize = band.YSize
            for i in range(0, xsize, tile_size_x):
                for j in range(0, ysize, tile_size_y):
                    com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                    os.system(com_string)

            
if __name__ == "__main__":
    get = "/split/"
    exp = "/split/splited/"
    split_raster(get, exp)
    print ('splitted')