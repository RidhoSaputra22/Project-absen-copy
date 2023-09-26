import pandas as pd
df = pd.read_csv('database/amd.csv')
name = 'Ridho '


ridho_row = df.loc[df['Nama'] == name]
print(ridho_row)

# Assuming you have the DataFrame and Ridho's row as shown in previous examples
ridho_row_str = ridho_row.to_csv(header=False, index=False).strip()

# Print the specific format
print(ridho_row_str)
