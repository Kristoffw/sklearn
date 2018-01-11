import tushare as ts


df = ts.get_tick_data('002625',date='2018-01-05')

print(df)