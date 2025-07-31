import pandas as pd

# 设置pandas显示选项，确保所有列都能完整显示
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 读取CSV文件
file_path = r'C:\Users\13106\Desktop\LLM-From-Scratch\dataset\train.csv'
df = pd.read_csv(file_path)

# 显示前几行数据
print("数据的前5行：")
print(df.head())

# 显示数据的基本信息
print("\n数据的基本信息：")
print(df.info())

# 显示数据的统计摘要
print("\n数据的统计摘要：")
print(df.describe())
