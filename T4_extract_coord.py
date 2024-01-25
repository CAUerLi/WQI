import os, time
import numpy as np
import pandas as pd

def main():
    os.chdir(r'F:/WQI20231129')
    Inputpath = 'MetaData'
    Outputpath = 'MetaData/unique_sites'
    filename = 'meta_TN_GRQA.xlsx'
    Syear = 1978
    Eyear = 2017
    Numyear = Eyear - Syear + 1
    resolution = 0.25
    OriData = pd.read_excel(os.path.join(Inputpath, filename), engine='openpyxl',header=0, index_col=0)  # nrows=500,
    TarData = OriData[(OriData['obs_year'] >= Syear) & (OriData['obs_year'] <= Eyear)].reset_index(drop=True)
    # TarData.to_excel('TarData.xlsx', index=True)
    statisticsData(TarData, Outputpath, 'unique_sites', 'coordinate')


def statisticsData(data, filepath, filename, txtfilename):

    unique_sites = data.drop_duplicates(subset='new_site_id', keep='first', inplace=False).reset_index(drop=True)
    unique_sites.loc[:, 'lat_wgs84'] = unique_sites['lat_wgs84'].apply(lambda x: f"{x:.2f}")  # .round(2)
    unique_sites.loc[:, 'lon_wgs84'] = unique_sites['lon_wgs84'].apply(lambda x: f"{x:.2f}")  # .round(2)
    # Group by the rounded 'lat_wgs84' and 'lon_wgs84' and take the first occurrence
    unique_sites_grouped = unique_sites.groupby(['lat_wgs84', 'lon_wgs84']).first().reset_index(drop=False)
    unique_sites_grouped = unique_sites_grouped.sort_values(by='new_site_id')
    unique_sites_grouped['new_new_site_id'] = range(1, len(unique_sites_grouped) + 1)
    # Save the unique sites to a new Excel file
    unique_sites_grouped.to_csv(os.path.join(filepath, filename + '.csv'), index=True)
    # Select the columns to save to the text file
    lat_lon_data = unique_sites_grouped[['lat_wgs84', 'lon_wgs84']]
    # Save the DataFrame to a text file, without headers and index
    lat_lon_data.to_csv(os.path.join(filepath, txtfilename + '.txt'), header=False, index=False, sep='\t')

    # # Convert the DataFrame to a CSV string format with tab as separator
    # lat_lon_data_str = unique_sites_grouped[['lat_wgs84', 'lon_wgs84']].to_csv(sep='\t', index=False, header=False)
    # # Define the path for the new txt file
    # # Write the string to a text file
    # with open(os.path.join(filepath, txtfilename + '.txt'), 'w') as f:
    #     f.write(lat_lon_data_str)
    # # Output the path to confirm where the file is saved
    print(f"Data saved")

### 这个地方是未来计算重复值用的
    # # 寻找重复值，将最后一个重复标记为True，寻找id
    # duplicates = unique_sites[unique_sites.duplicated(subset=['lat_wgs84', 'lon_wgs84'], keep=False)]
    # # Sort the duplicates to ensure we remove the correct rows after grouping
    # duplicates_sorted = duplicates.sort_values(by=['lat_wgs84', 'lon_wgs84', 'new_site_id'])
    # # 寻找到被重复id的name，便于后期处理。
    # removed_duplicates = duplicates_sorted.drop_duplicates(subset=['lat_wgs84', 'lon_wgs84'], keep='last')
    # # Save the removed duplicates to another Excel file
    # removed_duplicates_path = 'removed_duplicates_path.xlsx'
    # removed_duplicates.to_excel(removed_duplicates_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time taken: {elapsed_time: .2f}seconds')