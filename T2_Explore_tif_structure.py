###读取TIF文件可以用GDAL、Rioxarray和Rasterio  20240109

# import rioxarray
# pathSep = os.path.sep
# os.chdir('F:/data/Hydro90m/r.watershed/direction_tiles20d')
# tif_file_path = 'direction_h00v00.tif'
# data = rioxarray.open_rasterio(tif_file_path)
# print(type(data))
# # # 如果你想要处理或分析数据，可以直接使用xarray的功能
# # # 例如，计算数据的平均值
# mean = data.mean()
# print(f'Mean Value: {mean.values}')
# # 你还可以利用rioxarray的地理空间功能
# # 例如，重新投影数据
# reprojected_data = data.rio.reproject("EPSG:4326")
# # 查看重新投影后的数据
# print(reprojected_data)

import tqdm, os, time
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()
os.chdir('F:/WQI20231129')
def get_Tif_Info(tif_path):
    if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
        dataset = gdal.Open(tif_path)
        # osr.SpatialReference 提供描绘和转换坐标系统的服务 地理坐标(用经纬度表示)；投影坐标(如 UTM ，用米等单位量度来定位)。
        pcs = osr.SpatialReference()
        # ImportFromWkt()函数可以把 WKT坐标系统设置到OGRSpatialReference中
        pcs.ImportFromWkt(dataset.GetProjection())
        # pcs.SetWellKnownGeogCS("WGS84")
        gcs = pcs.CloneGeogCS()  # 地理空间坐标系
        band = dataset.GetRasterBand(1)
        no_data_value = band.GetNoDataValue()
        im_Geoetrans = dataset.GetGeoTransform()   # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        # 栅格数据的六参数。
        # geoTransform[0]：左上角像素经度
        # geoTransform[1]：影像宽度方向上的分辨率(经度范围/像素个数)
        # geoTransform[2]：x像素旋转, 0表示上面为北方
        # geoTransform[3]：左上角像素纬度
        # geoTransform[4]：y像素旋转, 0表示上面为北方
        # geoTransform[5]：影像宽度方向上的分辨率(纬度范围/像素个数)
        im_proj = dataset.GetProjection()  # 栅格数据的投影
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        shape = (im_bands, dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format"

    im_data = dataset.GetRasterBand(1).ReadAsArray()  # 写入数据中

    # img(ndarray), gdal数据集、地理空间坐标系、投影坐标系、栅格影像大小
    return im_data, dataset, no_data_value, im_Geoetrans, gcs, pcs, im_proj, im_width, im_height, im_bands, shape

def writeTif(newpath, im_data, im_Geoetrans, im_proj, datatype, compresstype, no_data_value):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName('Gtiff')
    creation_options = [compresstype]
    # creation_options = [
    #     "COMPRESS=LZW",  # 设置压缩类型为LZW
    #     "TILED=YES",  # 启用瓦片
    #     "BLOCKXSIZE=256",  # 设置X方向上的瓦片大小
    #     "BLOCKYSIZE=256"  # 设置Y方向上的瓦片大小
    # ]
    new_dataset = driver.Create(newpath, im_width, im_height, im_bands, datatype, options=creation_options)
    new_dataset.SetGeoTransform(im_Geoetrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        band = new_dataset.GetRasterBand(1)
        band.WriteArray(im_data)
        band.SetNoDataValue(no_data_value)
    else:
        for i in tqdm(range(im_bands)):
            band = new_dataset.GetRasterBand(i + 1)
            band.WriteArray(im_data[i])
            band.SetNoDataValue(no_data_value)
    del new_dataset

def write_asc(filepath, filename, array, xll, yll, cellsize, nodata=-9999):
    nrows, ncols = array.shape
    header = f"ncols        {ncols}\n"
    header += f"nrows        {nrows}\n"
    header += f"xllcorner    {xll}\n"
    header += f"yllcorner    {yll}\n"
    header += f"cellsize     {cellsize}\n"
    header += f"NODATA_value {nodata}\n"

    with open(os.path.join(filepath, filename), 'w') as f:
        f.write(header)
        np.savetxt(f, array, fmt="%d", delimiter=" ")

def main():
    Tifpath = r'F:/data/Hydro90m/r.watershed/direction_tiles20d'
    filename = os.path.join(Tifpath, 'direction_h28v04.tif')
    # newpath = r'F:/data/Hydro90m/r.watershed/direction_tiles20d/111.tif'
    im_data, dataset, no_data_value, im_Geoetrans, gcs, pcs, im_proj, \
        im_width, im_height, im_bands, shape = get_Tif_Info(filename)
    print(type(im_data))
    print(im_data.shape)
    print(im_data)
    print(im_proj)
    print(no_data_value)
    print(im_Geoetrans)
    print(im_bands)
    print(shape)

    # ncols = 'ncols         240000'
    # nrows = 'nrows         240000'
    # xllcorner = 'xllcorner     100'
    # yllcorner = 'yllcorner     25'
    # cellsize = 'cellsize      0.000833333334'
    # nodata_value = 'NODATA_value  -10'
    # header = '\n'.join([ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value])
    # np.savetxt(os.path.join(Tifpath, 'D_h28v04.asc'), im_data, header=header, comments=' ', fmt='%d')
    # write_asc(Tifpath, 'D_h28v04.asc', im_data, 100, 25, 0.000833333334, nodata=-10)

if __name__ == '__main__':
    start_time = time.time()
    main()
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

# writeTif(newpath, im_data, im_Geoetrans, im_proj, gdal.GDT_UInt16, 'COMPRESS=LZW')
# print(get_Tif_Info(filename))