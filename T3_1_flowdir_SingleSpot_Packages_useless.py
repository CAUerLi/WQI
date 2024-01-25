import numpy as np

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
    print("Running main function")

if __name__ == "__main__":
    main()