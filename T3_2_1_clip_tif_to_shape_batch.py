import os, time
from multiprocessing import Pool, cpu_count
from T3_2_clip_tif_to_shape_Packages import process_tiff_to_shp

os.chdir('F:/WQI20231129')

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]

def batchshapefile(ClipTif_path):

    output_directory = 'Output/MERIT_plus_05min/ShapeFile'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    coordinates = read_coordinates('MetaData/unique_sites_coordinate/coordinate.txt')
    coord_lists = [coord for coord in coordinates]
    # for coord in tqdm(coord_lists, desc="Processing items"):
    #     tiff_file_path = os.path.join(ClipTif_path, 'upstreamgrids' + '_' + str(coord[1]) + '_' + str(coord[0]) + '.tif')
    #     output_filename =  'shape_' + str(coord[1]) + '_' + str(coord[0])

    #     process_tiff_to_shp(tiff_file_path, output_directory, output_filename)

    args_list = [(os.path.join(ClipTif_path, 'upstreamgrids' + '_' + str(coord[1]) + '_' + str(coord[0]) + '.tif'),
                  output_directory, 'shape_' + str(coord[1]) + '_' + str(coord[0]))
                 for coord in coord_lists]
    # Utilize all available CPU cores
    pool = Pool(processes=cpu_count()-2)
    results = pool.map(process_tiff_to_shp, args_list)
    pool.close()
    pool.join()

if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists('error_logs'):
        os.makedirs('error_logs')
    ClipTif_path = os.path.join('Output', 'MERIT_plus_05min', 'Tiff')
    batchshapefile(ClipTif_path)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    