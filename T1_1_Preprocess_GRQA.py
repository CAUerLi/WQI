import os, time, math
import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import StandardScaler
pathSep = os.path.sep
os.chdir('F:/WQI20231129')

def cal_mean(df, id, year, TNvalue):  # for annual average
    datamean = df.groupby([id, year])[TNvalue].mean().reset_index()
    other_columns = df.groupby([id, year]).first().drop(['obs_date', 'obs_value'], axis=1).reset_index()
    df = other_columns.merge(datamean, on=[id, year], how='left').sort_values(by=[id, year], ascending=[True, True])
    return df

def split_date(df, obs_date):
    # df['year'] = df['obs_date'].str.extract(r'(\d{4})')
    df[obs_date] = pd.to_datetime(df[obs_date])
    df['obs_year'] = df['obs_date'].dt.year
    df['obs_month'] = df['obs_date'].dt.month
    return df

def extractdata(indir, vari):

    # Import data
    site_dtypes = {
        'lat_wgs84': np.float64,
        'lon_wgs84': np.float64,
        'obs_date': str,
        'site_id': str,
        'site_name': str,
        'site_country': str,
        'upstream_basin_area': str,
        'upstream_basin_area_unit': str,
        'param_code': str,  # object,
        'source_param_code': str,
        'param_name': str,
        'source_param_name': str,
        'obs_value': np.float64,
        'source_obs_value': np.float64,
        'unit': object,
        'filtration': object,
        'site_ts_availability': np.float64,
        'site_ts_continuity': np.float64,
        'obs_percentile': np.float64,
        'obs_iqr_outlier': object
    }

    # Metadata file
    source_file = os.path.join(indir, vari + '_' + 'GRQA' + '.csv')
    # sourcedf1 = pd.read_csv(source_file, sep=';')
    # sourcedf1.to_csv('origin_TN_GRQA.csv', index=True)
    sourcedf = pd.read_csv(source_file, sep=';', usecols=site_dtypes.keys(), dtype=site_dtypes)
    sourcedf['new_site_id'] = sourcedf.groupby(['lat_wgs84', 'lon_wgs84']).ngroup()
    metadf = split_date(sourcedf, 'obs_date')
    metadf = metadf.sort_values(by=['new_site_id', 'obs_year', 'obs_month'], ascending=[True, True, True])
    # cmap_df = cal_mean(cmap_df, 'new_site_id', 'year', 'obs_value')
    base_properties = ['new_site_id', 'site_id', 'site_name', 'lat_wgs84', 'lon_wgs84', 'site_country', 'obs_date', 'obs_year', 'obs_month']
    date_col = base_properties + [col for col in metadf.columns if col not in base_properties]
    metadf = metadf[date_col].reset_index(drop=True)

    # metadf.to_csv('Processing_meanTN_GRQA.csv', index=True)
    metadf.to_csv('meta_' + vari + '_GRQA.csv', index=True)

def main():
    # Specify the directory to start from
    root_dir = 'F:/data/GRQA/GRQA_data_v1.3'
    variables = []

    # Walk through all subdirectories and files in the directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if '_' in file:
                filesN = file.split('_', 1)
                variables.append(filesN[0])
    print(variables)
    print(len(variables))
    extractdata(root_dir, 'TN')



if __name__ == "__main__":

    start_time = time.time()
    main()
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print('first 2')
