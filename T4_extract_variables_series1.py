import os, time
import pandas as pd
from tqdm import tqdm
from osgeo import gdal

# Set up a working directory
os.chdir('F:/WQI20231129')
# Enable GDAL exceptions
gdal.UseExceptions()

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]
    
def read_tif(tif_path, variable_filename):

    dataset = gdal.Open(os.path.join(tif_path, variable_filename + '.tif'))
    geo_transform = dataset.GetGeoTransform()
    data = dataset.GetRasterBand(1).ReadAsArray()
    del dataset
    return geo_transform, data

def extract_value(tif_path, variable_filename, coord_list):
    output = {}
    geo_transform, data = read_tif(tif_path, variable_filename)
    print('start extracting ' + variable_filename)
    for coord in tqdm(coord_list):
        target_row = int((geo_transform[3] - coord[1]) / geo_transform[1])
        target_col = int((coord[0] - geo_transform[0]) / geo_transform[1])
        if 0 <= target_row < data.shape[0] and 0 <= target_col < data.shape[1]:
                output[coord] = data[target_row, target_col]
        else:
            output[coord] = None 
    return(output)               
        
def output_to_csv(variable_data, col_name, file_directory, file_name):
    # df = pd.DataFrame(list(variable_data.items()), columns=['Coordinates', 'Value'])
    df = pd.DataFrame([(coord[1], coord[0], value) for coord, value in list(variable_data.items())], 
                      columns=['Longitude', 'Latitude', col_name])
    df.to_csv(os.path.join(file_directory, file_name + '.csv'), index=False)

def main():
    
    coordinates = read_coordinates('MetaData/unique_sites_coordinate/coordinate.txt')
    tif_path = 'MetaData/HESS_MERITHydroIHU'
    variables = ['05min_rivlen', '05min_rivwth', '05min_rivlen_ds', '05min_rivslp', '05min_elevtn', '05min_strord', '05min_subare', '05min_uparea']
    for variable in tqdm(variables):
        dict_ext_value = extract_value(tif_path, variable, coordinates)  
        output_to_csv(dict_ext_value, variable.split('_')[1], 'Output', 'HESS' + variable)
  
if __name__ == "__main__":
    main()