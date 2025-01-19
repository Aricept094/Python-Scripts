import pandas as pd
import pygwalker as pyg

# Read Excel with optimized parameters
df = pd.read_excel('/home/aricept094/mydata/IVF.xlsx', engine='openpyxl')

# Enable kernel computation for better performance
walker = pyg.walk(df, kernel_computation=True)
