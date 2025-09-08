import pandas as pd 

file_path = "python_test.xlsx"
df = pd.read_excel(file_path, sheet_name="so",header=1)

df_clean = df[['date','product_category','so']].copy()
df_clean['date']= pd.to_datetime(df_clean['date']), errors ='coerce')
df_clean['month'] = df_clean['date'].dt.month # 月份数字

# Furniture and office supplies

# df_filtered = df_clean[df_clean['product_category']].str.contains("Furniture|Office Supplies",case = False,na=False)


# 合并table
def normalized_cat(x:str):
    if isintance(x,str):
        xl =x.lower()
        if 'furniture' in xl:
            return 'Furniure'
        if 'office supplies' in xl:
            return 'Office Supplies'

df_clean['big_cat'] = df_clean['product_category'].apply(normalized_cat)
df_clean = df_clean[df_clean['big_cat'].notna()]




# 表
# pivot_table =pd.pivot_table(
#     df_filtered,
#     index = "month",
#     columns = "product_category",
#     values ="so",
#     aggfunc ="sum",
#     fill_value=0

# )
pivot_table = (df_clean.groupby(['month','big_cat'])['so']
                        .sum()
                        .unstack('big_cat')
                        .fillna(0)
                        .astype((int))

)
 print(pivot_table)


# 按月度区分北京的每月的占比有个扩展
df_clean.loc

 

