import os
import time
import traceback
import numpy as np
import pandas as pd
from osgeo import gdal
from T2_Explore_tif_structure import writeTif
from T3_1_flowdir_SingleSpot_Packages import process_chunk
from multiprocessing import Pool, cpu_count

# 设置工作目录
os.chdir('F:/WQI20231129')

def export_to_excel(results, longitude, latitude):
    filename = f'upstreamgrids_{longitude}_{latitude}.xlsx'
    filepath = os.path.join('Output', 'xlsx', filename)
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df = pd.DataFrame(list(results.items()), columns=['coordinates', 'value'])
        df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
        df.drop('coordinates', axis=1, inplace=True)
        df.to_excel(writer, index=False)

def export_to_tiff(results, num_rows, num_cols, geo_transform, projection, longitude, latitude):
    filename = f'upstreamgrids_{longitude}_{latitude}.tif'
    filepath = os.path.join('Output', 'TIFF', filename)
    output_data = np.ones((num_rows, num_cols)) * 247
    for (local_i, j), value in results.items():
        output_data[local_i, j] = value
    writeTif(filepath, output_data, geo_transform, projection, gdal.GDT_UInt16, 'COMPRESS=LZW')

def process_point(args):
    # Unpack arguments
    (longitude, latitude), global_flow_direction, num_rows, num_cols, geo_transform, projection = args
    try:
        # Calculate target indices based on longitude and latitude
        target_row = int((85 - latitude) / 0.25)
        target_col = int((longitude + 180) / 0.25)

        # 划分数据块
        num_chunks = 2
        row_chunks = np.array_split(global_flow_direction, num_chunks)
        chunk_data = []
        start_row = 0
        target_chunk_index = None

        # for index, chunk in enumerate(row_chunks):
        #     chunk_data.append((chunk, start_row, target_row, target_col))
        #     if start_row > target_row and target_chunk_index is None:
        #         target_chunk_index = index - 1
        #     start_row += chunk.shape[0]

        for num, chunk in enumerate(row_chunks):
            chunk_data.append((chunk, start_row, target_row, target_col))
            if start_row > target_row and target_chunk_index is None:
                target_chunk_index = num - 1
                # break  # 或许跳过也可以..  基本上都是往后延一个
            start_row += chunk.shape[0]
            if target_chunk_index is None:
                target_chunk_index = num_chunks - 1
        # 处理特定块
        # Export results
        results = process_chunk(chunk_data, target_chunk_index)
        export_to_excel(results, longitude, latitude)
        export_to_tiff(results, num_rows, num_cols, geo_transform, projection, longitude, latitude)

        return f"Processed point ({longitude}, {latitude}) successfully."

    except Exception as e:
        error_message = f"Error processing point ({longitude}, {latitude}): {str(e)}\n{traceback.format_exc()}"
        error_file_path = f'error_logs/point_{longitude}_{latitude}.error'
        with open(error_file_path, 'w') as error_file:
            error_file.write(error_message)
        return error_message

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]
def main():
    # 加载数据
    tif_path = r'F:/data/MERIT-Plus_Dataset/MERIT_plus_15min_v2.2'
    flow_direction_filename = 'MERIT_plus_15min_v2.2_flwdir.tif'
    dataset = gdal.Open(os.path.join(tif_path, flow_direction_filename))
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    global_flow_direction = dataset.GetRasterBand(1).ReadAsArray()
    num_rows, num_cols = global_flow_direction.shape
    del dataset

    # Read coordinates from file
    coordinates = read_coordinates('MetaData/unique_sites_coordinate/coordinate.txt')
    # Prepare arguments for multiprocessing
    args_list = [(coord, global_flow_direction, num_rows, num_cols, geo_transform, projection)
                 for coord in coordinates]
    # Utilize all available CPU cores
    pool = Pool(processes=cpu_count()-2)
    results = pool.map(process_point, args_list)
    pool.close()
    pool.join()

    for result in results:
        print(result)


if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists('error_logs'):
        os.makedirs('error_logs')
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
