import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your CSV file
csv_file_path = "C:/Users/syslab123/Desktop/Act to Reason - Original/Act-to-reason-original/experiments/some_title/dynamic/training/training_data/ego.csv"

# Read the CSV file
df = pd.read_csv(csv_file_path)
print(df.size)
# Extract the third to last column (excluding the header)

third_last_column = df.iloc[:, -3]

third_last_column = third_last_column[60000:60200]
print("size of the last 3rd column is: " + str(third_last_column.size))

# Create an index for the x-axis (excluding the header row)
x_axis = range(1, len(third_last_column) + 1)

LEV1 = np.zeros(third_last_column.size)
LEV2 = np.zeros(third_last_column.size)
LEV3 = np.zeros(third_last_column.size)


LEV1[third_last_column == 0] = 1
LEV2[third_last_column == 1] = 2
LEV3 [third_last_column == 2] = 3


print (LEV1.size)
print (LEV2.size)
print (LEV3.size)

plt.plot(LEV1 , '^', label='level = 1', markersize=1)
plt.plot(LEV2,  '^', label='level = 2', markersize=1)
plt.plot(LEV3,  '^', label='level = 3', markersize=1)

# Add labels and title
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.title('Plot of Third to Last Column Values')
plt.legend()
plt.show()


