import os, time, math
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma
from sklearn.preprocessing import StandardScaler

import glob
# import hashlib #在Python中, hashlib模块提供了各种哈希算法的实现，包括常见的MD5、SHA-1、SHA-256等
# 您可以使用这些哈希算法来对数据进行加密、验证或创建数字签名等操作。

def cal_mean(df, id, year, TNvalue):

    datamean = df.groupby([id, year])[TNvalue].mean().reset_index()
    other_columns = df.groupby([id, year]).first().drop(['obs_date', 'obs_value'], axis=1).reset_index()
    df = other_columns.merge(datamean, on=[id, year], how='left').sort_values(by= [id, year], ascending=[True, True])
    return df

def split_year(df, obs_date):

    # df['year'] = df['obs_date'].str.extract(r'(\d{4})')
    df[obs_date] = pd.to_datetime(df[obs_date])
    df['year'] = df['obs_date'].dt.year
    return df

def gwqdata():

    # Directory paths
    proj_dir = 'NCdata'
    proj_name = 'TN_GRQA'

    # Import data
    site_dtypes = {
        'lat_wgs84': np.float64,
        'lon_wgs84': np.float64,
        'obs_date': object,
        'site_id': object,
        'site_name': str,
        'site_country': str,
        'upstream_basin_area': np.float64,
        'upstream_basin_area_unit': str,
        'param_code': object,
        'param_name': object,
        'obs_value': np.float64,
        'unit': object,
        'filtration': object,
        'site_ts_availability': object,
        'site_ts_continuity': object
    }
    # Metadata file
    meta_file = os.path.join(proj_dir, proj_name + '.csv')
    cmap_df = pd.read_csv(meta_file, sep=';', usecols=site_dtypes.keys(), dtype=site_dtypes)
    cmap_df['new_site_id'] = cmap_df.groupby(['lat_wgs84', 'lon_wgs84']).ngroup()

    cmap_df = split_year(cmap_df, 'obs_date')
    cmap_df = cal_mean(cmap_df, 'new_site_id', 'year', 'obs_value')

    date_col = ['new_site_id', 'lat_wgs84', 'lon_wgs84', 'year', 'param_code', 'obs_value', 'unit'] \
               + [col for col in cmap_df.columns if col not in ['new_site_id', 'lat_wgs84', 'lon_wgs84', 'year', 'param_code', 'obs_value', 'unit']]

    out_df = cmap_df[date_col]

    # cmap_df.to_csv('Processing_meanTN_GRQA.csv', index=False)
    return out_df

def read_ncdata(path, name, cVari):
    ncdata = Dataset(os.path.join(path, name))
    climateDU = ncdata.variables[cVari]
    fillV = climateDU[0].fill_value
    climateD = ma.filled(climateDU)  # [:,:,:]
    climateD[climateD == fillV] = np.nan
    climateD[climateD < 0] = np.nan
    return climateD

def pepic_totnwdata():  # 这个是将三种作物两种种植方式进行进行面积的加权平均

    scenario = 'AgMERRA-51-60'
    startY = 1980
    endY = 2010
    year = str(startY) + '_' + str(endY)
    numY = endY - startY + 1
    vairs = 'nWater'
    scen = 'PrepN-' + scenario


    shape = (numY, 360, 720)
    dataAnn = np.full(shape, -9999)

    shape1 = (360, 720)
    data_area = np.full(shape1, -9999)

    maize_ir = read_ncdata('NCdata', 'mai' + scen + '_Maize_IR_' + vairs + '_' + year + '.nc4', vairs)
    maize_rf = read_ncdata('NCdata', 'mai' + scen + '_Maize_RF_' + vairs + '_' + year + '.nc4', vairs)
    rice_ir = read_ncdata('NCdata', 'ric' + scen + '_Rice_IR_' + vairs + '_' + year + '.nc4', vairs)
    rice_rf = read_ncdata('NCdata', 'ric' + scen + '_Rice_RF_' + vairs + '_' + year + '.nc4', vairs)
    wheat_ir = read_ncdata('NCdata', 'whe' + scen + '_Wheat_IR_' + vairs + '_' + year + '.nc4', vairs)
    wheat_rf = read_ncdata('NCdata', 'whe' + scen + '_Wheat_RF_' + vairs + '_' + year + '.nc4', vairs)

    maize_ir_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Maize' + '_' + 'IR' + '.asc'), skip_header=6)
    maize_rf_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Maize' + '_' + 'RF' + '.asc'), skip_header=6)
    rice_ir_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Rice' + '_' + 'IR' + '.asc'), skip_header=6)
    rice_rf_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Rice' + '_' + 'RF' + '.asc'), skip_header=6)
    wheat_ir_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Wheat' + '_' + 'IR' + '.asc'), skip_header=6)
    wheat_rf_landuse = np.genfromtxt(os.path.join('NCdata', 'SPAM2010_HarArea_Wheat' + '_' + 'RF' + '.asc'), skip_header=6)

    for iY in range(numY):
        for m in range(dataAnn.shape[1]):
            for n in range(dataAnn.shape[2]):
                nwaterdt = []
                nwaterdt.append(maize_ir[iY, m, n])
                nwaterdt.append(maize_rf[iY, m, n])
                nwaterdt.append(rice_ir[iY, m, n])
                nwaterdt.append(rice_rf[iY, m, n])
                nwaterdt.append(wheat_ir[iY, m, n])
                nwaterdt.append(wheat_rf[iY, m, n])
                area = []
                area.append(maize_ir_landuse[m, n])
                area.append(maize_rf_landuse[m, n])
                area.append(rice_ir_landuse[m, n])
                area.append(rice_rf_landuse[m, n])
                area.append(wheat_ir_landuse[m, n])
                area.append(wheat_rf_landuse[m, n])
                if not all(math.isnan(item) for item in nwaterdt): # and not all(it <= 0 for it in area)
                    dataAnn[iY, m, n] = weighted_average(nwaterdt, area)
                if iY == numY-1:
                    if any(im > 0 for im in area):
                        data_area[m, n] = np.nansum(area)

    np.savetxt('temp.men', data_area, fmt='%.9f')
    fOut = open('temp.men', 'r')
    outTemp = fOut.readlines()
    fOut.close()
    os.remove('temp.men')
    fOut = open(os.path.join('NCdata', 'ToT_crop_area'+ '.asc'), 'w')
    fOut.write('ncols         720\n')
    fOut.write('nrows         360\n')
    fOut.write('xllcorner     -180\n')
    fOut.write('yllcorner     -90\n')
    fOut.write('cellsize      0.5\n')
    fOut.write('NODATA_value  -9999\n')
    fOut.writelines(outTemp)
    fOut.close()
    generateNC(dataAnn, 'TOT', 'NCdata', 'nwater', startY, endY)

def weighted_average(list_a, list_b):  # 因为6种方式 这是加权平均的核心公式

    # 初始化变量
    total_weight = 0
    weighted_sum = 0

    for a, weight in zip(list_a, list_b):
        if not math.isnan(a):
            weighted_sum += a * weight
        if weight >= 0:
            total_weight += weight
    if total_weight == 0:
        return -9999  # 避免除以零的情况
    else:
        return weighted_sum / total_weight

def generateNC(data, crop, ncFilePath, variable, startYear, endYear):
    ### generate the NC file for specific variable
    pathSep = os.path.sep
    ### define the study area domain
    fStudyAreaText = open('NCdata' + pathSep + 'fpu.asc', 'r')
    colNum = int(fStudyAreaText.readline().strip(os.linesep).split()[1])
    rowNum = int(fStudyAreaText.readline().strip(os.linesep).split()[1])
    xLeftCor = float(fStudyAreaText.readline().strip(os.linesep).split()[1])
    yBottomCor = float(fStudyAreaText.readline().strip(os.linesep).split()[1])
    resolution = float(fStudyAreaText.readline().strip(os.linesep).split()[1])
    fStudyAreaText.close()
    xRightCor = xLeftCor + resolution * colNum
    yUpCor = yBottomCor + resolution * rowNum
    ### prepare NC file
    ncFileName = crop + '_' + variable + '_' + str(startYear) + '_' + str(
        endYear) + '.nc4'
    root_Data = Dataset(ncFilePath + pathSep + ncFileName, 'w', format='NETCDF4', zlib=True, complevel=9,
                        least_significant_digit=2)
    t = [i for i in range(1, endYear - startYear + 2)]
    ### create dimension
    root_Data.createDimension('lon', colNum)
    root_Data.createDimension('lat', rowNum)
    root_Data.createDimension('time', None)
    ### create variables
    times = root_Data.createVariable('time', 'f8', ('time'))
    latitudes = root_Data.createVariable('lat', 'f4', ('lat'))
    longitudes = root_Data.createVariable('lon', 'f4', ('lon'))
    dataNC = root_Data.createVariable(variable, 'f4', ('time', 'lat', 'lon'), fill_value='1.e+20', zlib=True,
                                      complevel=9)
    ### set attributes
    root_Data.description = 'PEPIC simulated ' + variable + ' between ' + str(startYear) + ' and ' + str(endYear)
    root_Data.history = 'Created ' + time.ctime(time.time()) + ' By Wenfeng Liu'
    root_Data.institution = 'LSCE & Eawag'
    root_Data.contact = 'Mengxue Li <lmxx569@163.com>'
    dataNC._DeflateLevel = '9'
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    times.units = 'growing seasons since ' + str(startYear) + '-01-01 00:00:00'
    times.calendar = 'gregorian'
    ### write variable
    lats = np.arange(yUpCor - resolution / 2, yBottomCor, -resolution)
    lons = np.arange(xLeftCor + resolution / 2, xRightCor, resolution)
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = t
    for n in t:
        mask = data[n - t[0], :, :] < 0
        inData = np.ma.MaskedArray(data[n - t[0], :, :], mask=mask, fill_value=1.e+20, dtype=np.float64)
        dataNC[n - t[0], :, :] = inData
        del inData
    root_Data.close()

# pepic_totnwdata()
# waterdata = gwqdata()

waterdata = pd.read_csv(os.path.join('water quality data', 'Processing_meanTN_GRQA' + '.csv'), header=0)
PEPICData = Dataset(os.path.join('NCdata', 'TOT_nwater_1980_2010' + '.nc4'))
nwaterPEPIC = PEPICData.variables['nwater'][:]

for m in range(waterdata.shape[0]):
    # for iY in range(waterdata.shape[1]):
    siteid = [[waterdata.loc[m, 'lat_wgs84'], waterdata.loc[m, 'lon_wgs84']]]
    if siteid[0][0] >= -89.75:
        Row = int((90 - siteid[0][0] + 0.25) / 0.5)
    else:
        Row = 359
    if siteid[0][1] <= 179.75:
        Col = int((180 + siteid[0][1] - 0.25) / 0.5)
    else:
        Col = 719
    if 1980 <= waterdata.loc[m, 'year'] <= 2010:
        nw = nwaterPEPIC[int(waterdata.loc[m, 'year'] - 1980), Row, Col]
        if nw != None:
            waterdata.loc[m, 'PEPIC_nW'] = nw

# cStat = waterdata['PEPIC_nW'] > 0
# outStat = waterdata[cStat].reset_index(drop=True)
outStat_df = waterdata[waterdata['PEPIC_nW'].notna()].reset_index(drop=True)

grouped = outStat_df.groupby('new_site_id')
# 这里我仅仅选取了年数量大于5的
outStat_df['count'] = grouped['new_site_id'].transform('count')
outStat_df = outStat_df[outStat_df['count'] >= 5]
outStat_df = outStat_df.drop(columns=['count'])

scaler = StandardScaler()  # 创建一个标准化器
outStat_df['WQD_normalized'] = grouped['obs_value'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel())
outStat_df['PEPIC_normalized'] = grouped['PEPIC_nW'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel())

outStat_df.to_csv('water quality data/Processing_GRQA_final_ver1.csv', index=False)

# 获取唯一的 site_num 值
# site_num = outStat_df['new_site_id'].tolist()
# unique_site_num = list(set(site_num))
unique_site_num = outStat_df['new_site_id'].unique()

cor_dic = {}
for num in unique_site_num:
    subset_df = outStat_df[outStat_df['new_site_id'] == num]
    WQD = subset_df['WQD_normalized'].tolist()
    PEPICnw = subset_df['PEPIC_normalized'].tolist()
    correlation_coefficient = np.corrcoef(WQD, PEPICnw)[0, 1]
    cor_dic[num] = correlation_coefficient

correlation_df = pd.DataFrame(cor_dic.items(), columns=['new_site_id', 'correlation_coefficient'])
correlation_df.to_csv('water quality data/correlation_between_grid.csv', index=False)























