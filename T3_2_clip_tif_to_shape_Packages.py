import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import os

def process_tiff_to_shp(args): #(tiff_file_path, output_directory, output_filename):
    # 使用 Rasterio 读取 TIFF 文件
    tiff_file_path, output_directory, output_filename = args
    with rasterio.open(tiff_file_path) as src:
        data = src.read(1)  # 读取第一个波段
        nodata_value = src.nodata  # NoData 值

        # 创建掩膜，排除 NoData 值的区域
        mask = data != nodata_value

        # 提取非 NoData 区域的几何形状
        # 使用 shapes 函数提取非 NoData 区域的几何形状
        results = ({'properties': {'raster_val': v}, 'geometry': s}
                   for i, (s, v) in enumerate(shapes(data, mask=mask, transform=src.transform)))

    # 创建 GeoDataFrame
    geoms = [shape(geom['geometry']) for geom in results]
    gdf = gpd.GeoDataFrame({"geometry": geoms})

    # 设置坐标参考系统为 WGS 84 (EPSG:4326)
    gdf.crs = "EPSG:4326"

    # gdf.to_file(filename=output_directory, driver='ESRI Shapefile')
    # 保存 SHP 文件集
    # 这里虽然是加的'.shp',但是Geopandas的.to_file会生成所有的文件类型
    output_shp_path = os.path.join(output_directory, output_filename + ".shp")

    # 保存 SHP 文件集
    gdf.to_file(filename=output_shp_path, driver='ESRI Shapefile')

'''
# 定义文件路径
tiff_file_path = 'F:/WQI20231129/WQI/chunknum_11_YellowRiver15arcmin.tif'  # 替换为您的 TIFF 文件路径
output_directory = 'F:/WQI20231129/Shape'  # 指定输出目录
output_filename = 'desired_output_name'  # 指定输出文件名

# 调用函数处理 TIFF 文件并生成 SHP 文件
process_tiff_to_shp(tiff_file_path, output_directory, output_filename)

# 确认生成的文件
shp_files = os.listdir(output_directory)
print(shp_files)
'''

def main():
    print("Running main function")

if __name__ == "__main__":
    main()