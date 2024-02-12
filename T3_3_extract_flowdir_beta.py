import os
import time
import traceback
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
from T2_Explore_tif_structure import writeTif
from T3_2_clip_tif_to_shape_Packages import process_tiff_to_shp
from multiprocessing import Pool, cpu_count

# Set up a working directory
os.chdir('F:/WQI20231129')

def up_point(chunk_dir, i, j, numcol): 
    dirOut = []
    if chunk_dir[i, j] >= 0:
        if j - 1 >= 0:
            if chunk_dir[i - 1, j - 1] == 2: dirOut.append((i - 1, j - 1))
            if chunk_dir[i, j - 1] == 1: dirOut.append((i, j - 1))
            if chunk_dir[i + 1, j - 1] == 128: dirOut.append((i + 1, j - 1))
        if j + 1 < numcol:
            if chunk_dir[i - 1, j + 1] == 8: dirOut.append((i - 1, j + 1))
            if chunk_dir[i, j + 1] == 16: dirOut.append((i, j + 1))
            if chunk_dir[i + 1, j + 1] == 32: dirOut.append((i + 1, j + 1))
        if chunk_dir[i - 1, j] == 4: dirOut.append((i - 1, j))
        if chunk_dir[i + 1, j] == 64: dirOut.append((i + 1, j))
    return dirOut

def export_to_excel(results, flow_direction_filename, longitude, latitude):
    filedirct = os.path.join('Output', flow_direction_filename, 'Xlsx')
    if not os.path.exists(filedirct):
        os.makedirs(filedirct)
    filename = f'upstreamgrids_{latitude}_{longitude}.xlsx'
    filepath = os.path.join(filedirct, filename)
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df = pd.DataFrame(list(results.items()), columns=['coordinates', 'value'])
        df[['latitude', 'longitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
        df.drop('coordinates', axis=1, inplace=True)
        df.to_excel(writer, index=False)

def export_to_tiff(results, num_rows, num_cols, geo_transform, projection, longitude, latitude, flow_direction_filename):
    filedirct = os.path.join('Output', flow_direction_filename, 'Tiff')
    if not os.path.exists(filedirct):
        os.makedirs(filedirct)
    filename = f'upstreamgrids_{latitude}_{longitude}.tif'
    filepath = os.path.join(filedirct, filename)
    output_data = np.ones((num_rows, num_cols)) * -9999
    for (i, j), value in results.items():
        output_data[i, j] = value
    writeTif(filepath, output_data, geo_transform, projection, gdal.GDT_Int16, 'COMPRESS=LZW', no_data_value=-9999)

def tot_up_point(dir_data, i, j, numcol):
    res = {}
    stack = [(i, j, 1)]

    while stack:
        Ui, Uj, ic = stack.pop()
        if (Ui, Uj) not in res:
            res[(Ui, Uj)] = ic
            up_points = up_point(dir_data, Ui, Uj, numcol)
            for pt in up_points:
                stack.append((*pt, ic + 1))
    return res

def process_point(args):
    # Unpack arguments
    (longitude, latitude), global_flow_direction, num_rows, num_cols, geo_transform, projection, flow_direction_filename = args
    try:
        # Calculate target indices based on longitude and latitude
        target_row = int((geo_transform[3] - latitude) / geo_transform[1])
        target_col = int((longitude - geo_transform[0]) / geo_transform[1])
        
        # Export results
        result = tot_up_point(global_flow_direction, target_row, target_col, num_cols)
        
        export_to_excel(result, flow_direction_filename, longitude, latitude)
        export_to_tiff(result, num_rows, num_cols, geo_transform, projection, longitude, latitude, flow_direction_filename)

        return f"Processed point ({latitude}, {longitude}) successfully."

    except Exception as e:
        error_message = f"Error processing point ({latitude}_{longitude}): {str(e)}\n{traceback.format_exc()}"
        error_file_path = f'error_logs/point_{latitude}_{longitude}.error'
        with open(error_file_path, 'w') as error_file:
            error_file.write(error_message)
        return error_message

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]

def batchShapefile(flow_direction_filename, coordinates):
    ClipTif_directory = os.path.join('Output', flow_direction_filename, 'Tiff')
    output_directory = os.path.join('Output', flow_direction_filename, 'ShapeFile')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    coord_lists = [coord for coord in coordinates]

    args_list = [(os.path.join(ClipTif_directory, 'upstreamgrids' + '_' + str(coord[1]) + '_' + str(coord[0]) + '.tif'),
                  output_directory, 'shape_' + str(coord[1]) + '_' + str(coord[0]))
                 for coord in coord_lists]
    #### Utilize all available CPU cores
    # pool = Pool(processes=cpu_count()-2)
    # results = pool.imap(process_tiff_to_shp, args_list)
    # for result in tqdm(results):
    #     result = result
    # pool.close()
    # pool.join()
    print('start converting to Shapefile format')
    with Pool(processes=cpu_count()-2) as pool:
        results = pool.imap(process_tiff_to_shp, args_list)
        for result in tqdm(results, total = len(coordinates)):
            pass
    
def batchTiffile(tif_path, flow_direction_filename):  # , coordinates
    # load data
    dataset = gdal.Open(os.path.join(tif_path, flow_direction_filename + '.tif'))
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    global_flow_direction = dataset.GetRasterBand(1).ReadAsArray()
    num_rows, num_cols = global_flow_direction.shape
    del dataset

    # Prepare arguments for multiprocessing
    args_list = [(coord, global_flow_direction, num_rows, num_cols, geo_transform, projection, flow_direction_filename)
                 for coord in coordinates]
    
    print('start generating Tif and Excel files')
    # Utilize all available CPU cores
    with Pool(processes=cpu_count()-2) as pool:
        results = pool.imap(process_point, args_list)
        for result in tqdm(results, total = len(coordinates)):
            pass

if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists('error_logs'):
        os.makedirs('error_logs')
    tif_directory = 'F:/WQI20231129/MetaData/FlowDirDateset'
    flow_direction_filename = 'HESS_05min_flwdir' # 'hyd_glo_dir_5m' MERIT_plus_05min_v2.2_flwdir'
    # Read coordinates from file
    coordinates = read_coordinates('MetaData/unique_sites_coordinate/coordinate.txt')
    batchTiffile(tif_directory, flow_direction_filename) # , coordinates
    batchShapefile(flow_direction_filename, coordinates)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")