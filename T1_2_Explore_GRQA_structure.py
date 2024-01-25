import os, time
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
pathSep = os.path.sep
# os.chdir('F:/WQI20231129')

def main():
    # 加载数据集
    file_path = 'F:/WQI20231129'
    WQI_data = pd.read_csv(os.path.join(file_path, 'meta_TN_GRQA' + '.csv'), header=0, index_col=0)
    propotion(WQI_data, file_path)
    select_year(WQI_data)

def propotion(WQI_data, file_path):
    # 计算每个年份和月份的出现比例
    year_proportions = WQI_data['obs_year'].value_counts(normalize=True).sort_index()
    month_proportions = WQI_data['obs_month'].value_counts(normalize=True).sort_index()

    year_proportions.to_csv(os.path.join(file_path, 'year' + '.csv'))
    month_proportions.to_csv(os.path.join(file_path, 'month' + '.csv'))

    # 绘制分布图
    plt.figure(figsize=(15, 6))

    # 年份分布图
    plt.subplot(1, 2, 1)
    sns.barplot(x=year_proportions.index, y=year_proportions.values)
    plt.title('Distribution of Observations by Year')
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)

    # 月份分布图
    plt.subplot(1, 2, 2)
    sns.barplot(x=month_proportions.index, y=month_proportions.values)
    plt.title('Distribution of Observations by Month')
    plt.xlabel('Month')
    plt.ylabel('Proportion')

    # 调整布局
    plt.tight_layout()
    # plt.show()
    plt.savefig('figure' + pathSep + 'year&mon_distribution' + '.jpg', format='JPG', dpi=600)


def select_year(WQI_data):
    # 计算每个年份的出现比例
    year_proportions = WQI_data['obs_year'].value_counts(normalize=True).sort_index()

    # 将年份比例转换为DataFrame，以便进行处理
    year_proportions_df = year_proportions.reset_index()
    year_proportions_df.columns = ['Year', 'Proportion']

    # 初始化一个DataFrame来存储结果
    results_df = pd.DataFrame(columns=['Start Year', 'End Year', 'Total Proportion'])

    # 循环计算每个连续30年时间段的总占比
    for i in trange(len(year_proportions_df) - 29):  # leave=True, position=0
        start_year = year_proportions_df['Year'][i]
        end_year = year_proportions_df['Year'][i + 29]
        total_proportion = year_proportions_df['Proportion'][i:i + 30].sum()
        new_row = pd.DataFrame({'Start Year': [start_year],
                                'End Year': [end_year],
                                'Total Proportion': [total_proportion]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 按总占比降序排序结果
    sorted_results = results_df.sort_values(by='Total Proportion', ascending=False).reset_index(drop=True)

    # 可视化前10个结果
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Total Proportion', y='Start Year', data=sorted_results.head(10), orient='h')
    plt.title('Top 10 30-Year Periods by Total Data Proportion')
    plt.xlabel('Total Proportion')
    plt.ylabel('Start Year of 30-Year Period')
    plt.savefig('figure' + pathSep + '30year_max' + '.jpg', format='JPG', dpi=600)


if __name__ == '__main__':

    start_time = time.time()
    main()
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    # print("Total time: {:.2f} seconds".format(elapsed_time))

