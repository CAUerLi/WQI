import geopandas as gpd
import matplotlib.pyplot as plt

# 读取 Shapefile 文件
# 替换为您的 Shapefile 文件路径
gdf = gpd.read_file(r'C:/Users/Mengxue Li/Desktop/New folder/shp_output.shp')

# 查看数据前几行
print(gdf.head())

# 绘制所有地理对象
gdf.plot()

# 显示绘制的地图
plt.show()