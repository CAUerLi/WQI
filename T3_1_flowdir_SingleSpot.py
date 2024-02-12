import os, time
import numpy as np
import pandas as pd
from osgeo import gdal
from T2_Explore_tif_structure import  writeTif, write_asc, get_Tif_Info

def up_point(chunk_dir, i, j, numcol = 1440):
    dirOut = []
    numrow = chunk_dir.shape[0]

    if chunk_dir[i, j] >= 0:
        if i - 1 >= 0 and j - 1 >= 0:
            if chunk_dir[i - 1, j - 1] == 2: dirOut.append((i - 1, j - 1))
        if i - 1 >= 0:
            if chunk_dir[i - 1, j] == 4: dirOut.append((i - 1, j))
        if i - 1 >= 0 and j + 1 < numcol:
            if chunk_dir[i - 1, j + 1] == 8: dirOut.append((i - 1, j + 1))
        if j - 1 >= 0:
            if chunk_dir[i, j - 1] == 1: dirOut.append((i, j - 1))
        if j + 1 < numcol:
            if chunk_dir[i, j + 1] == 16: dirOut.append((i, j + 1))
        if i + 1 < numrow and j - 1 >= 0:
            if chunk_dir[i + 1, j - 1] == 128: dirOut.append((i + 1, j - 1))
        if i + 1 < numrow:
            if chunk_dir[i + 1, j] == 64: dirOut.append((i + 1, j))
        if i + 1 < numrow and j + 1 < numcol:
            if chunk_dir[i + 1, j + 1] == 32: dirOut.append((i + 1, j + 1))
    return dirOut


def tot_up_point(chunk_dir, global_start_row, local_i, j):
    border_points = []
    res = {}
    stack = [(local_i, j, 1)]

    if is_on_border((local_i, j), chunk_dir.shape[0]):
        border_points.append((*(local_i, j), 0))
    else:
        while stack:
            Ui, Uj, ic = stack.pop()
            if (Ui, Uj) not in res:
                res[(Ui + global_start_row, Uj)] = ic
                up_points = up_point(chunk_dir, Ui, Uj)
                for pt in up_points:
                    if is_on_border(pt, chunk_dir.shape[0]):
                        border_points.append((*pt, ic + 1))  # return local coord
                    else:
                        stack.append((*pt, ic + 1))
    return res, border_points
def is_on_border(point, numrow):
    i, j = point
    return i == 0 or i == numrow - 1 # or j == 0 or j == numcol - 1
def process_adjacent_tot_up_point(chunk_dir, global_start_row, local_Ui, local_Uj, si):
    res = {}
    stack = [(local_Ui, local_Uj, si)]

    while stack:
        i, j, ic = stack.pop()
        if (i, j) not in res:
            res[(i + global_start_row, j)] = ic
            up_points = up_point(chunk_dir, i, j)
            for pt in up_points:
                stack.append((*pt, ic + 1))
    return res

def process_chunk(chunk_data, chunk_num):

    chunk_dir, global_start_row, Oi, Oj = chunk_data[chunk_num]
    local_Oi = Oi - global_start_row # 从0开始

    result_dir, border_dir = tot_up_point(chunk_dir, global_start_row, local_Oi, Oj)
    adjacent_result_dirs = {}
    for border_point in border_dir:  #  border_point = (local_i, j, ic)
        if border_point[0] == 0:
            # adjacent_chunks_num = chunk_num - 1
            combine_chunk_dir, global_start_bor_row, local_bor_i, local_bor_j, = \
                (np.vstack((chunk_data[chunk_num-1][0], chunk_data[chunk_num][0])), chunk_data[chunk_num-1][1],
                 chunk_data[chunk_num][1] - chunk_data[chunk_num-1][1], border_point[1])
            adjacent_result_dir = process_adjacent_tot_up_point(combine_chunk_dir, global_start_bor_row, local_bor_i, local_bor_j, border_point[2])
        else:
            # adjacent_chunks_num = chunk_num + 1
            combine_chunk_dir, global_start_bor_row, local_bor_i, local_bor_j = \
                (np.vstack((chunk_data[chunk_num][0], chunk_data[chunk_num + 1][0])), chunk_data[chunk_num][1],
                 border_point[0], border_point[1])
            adjacent_result_dir = process_adjacent_tot_up_point(combine_chunk_dir, global_start_bor_row, local_bor_i, local_bor_j, border_point[2])
        adjacent_result_dirs.update(adjacent_result_dir)
    result_dir.update(adjacent_result_dirs)
    return result_dir

def main():
    os.chdir('F:/WQI20231129')
    lon = 50
    lat = 3
    # for ii in range(2,16):
        
    Ti = int((85 - lat) / 0.25)
    Tj = int((lon + 180) / 0.25)

    tifpath = r'F:/data/MERIT-Plus_Dataset/MERIT_plus_15min_v2.2'
    filename = 'MERIT_plus_15min_v2.2_flwdir.tif' #'HESS_05min_flwdir.tif, MERIT_plus_05min_v2.2_flwdir
    dataset = gdal.Open(os.path.join(tifpath, filename))
    im_Geoetrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    global_dir = dataset.GetRasterBand(1).ReadAsArray()
    numRow, numCol = global_dir.shape
    del dataset

    # ascpath = r'F:/data/MERIT-Plus_Dataset/MERIT_plus_05min_v2.2'
    # global_dir = np.genfromtxt(os.path.join(ascpath, 'MERIT_plus_05min_v2.2_flwdir.asc'), skip_header=6)
    # numRow, numCol = global_dir.shape

    # Tif_Inf = get_Tif_Info(os.path.join(tifpath, filename))
    # print(Tif_Inf)

    num_chunks = 2
    row_chunks = np.array_split(global_dir, num_chunks)  # split dataset to different cu
    chunk_data = []
    start_row = 0
    chunk_num = None  # 判别位于哪个chunk中
    # 这里需要增加 判断num不为0
    for num, chunk in enumerate(row_chunks):
        chunk_data.append((chunk, start_row, Ti, Tj))
        if start_row > Ti and chunk_num is None:
            chunk_num = num - 1
            # break  # 或许跳过也可以..  基本上都是往后延一个
        start_row += chunk.shape[0]
        if chunk_num is None:
            chunk_num = num_chunks - 1
    results = process_chunk(chunk_data, chunk_num)

    file_path = 'F:/WQI20231129/WQI/num_chunks_' + str(num_chunks) + '_0118_3.xlsx'
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df = pd.DataFrame(list(results.items()), columns=['x_y', 'value'])
        df[['x', 'y']] = pd.DataFrame(df['x_y'].tolist(), index=df.index)
        df.drop('x_y', axis=1, inplace=True)
        # sheet_name = f'Sheet{i + 1}'
        df.to_excel(writer, index=False)

    dataOut = np.ones((numRow, numCol)) * 247
    for (local_i, j), val in results.items():
        dataOut[local_i, j] = val
    # write_asc('F:/WQI20231129/WQI', 'chunknum_' + str(num_chunks) +'_YellowRiver15arcmin.asc', dataOut, -180, -60, 0.25, nodata=-99)
    writeTif(os.path.join('F:/WQI20231129/WQI','chunknum_' + str(num_chunks) + '_YellowRiver15arcmin3.tif'), dataOut, im_Geoetrans, im_proj, gdal.GDT_UInt16, 'COMPRESS=LZW')

if __name__ == "__main__" :
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")