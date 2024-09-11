# Loan Data Processing Project
I recently completed a project for processing financial loan data using Python and NumPy. The following code illustrates the key steps I took to clean and preprocess the dataset.
## Code Overview
### Import Libraries and Configure Settings

```python
import numpy as np
np.set_printoptions(suppress=True, precision=2)
```

I set the print options for better readability and processed the loan dataset and then calculated the number of missing values, determined a temporary fill value, and computed the column means. Finally, I compiled an array of statistics, including the minimum, mean, and maximum values for each column, while ignoring missing data.
```python
np.set_printoptions(suppress=True, precision=2)
np.isnan(raw_data_np).sum()
temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis = 0)
temporary_stats = np.array([np.nanmin(raw_data_np, axis = 0),
                           temporary_mean,
                           np.nanmax(raw_data_np, axis = 0)])
```

In the next step, I identified which columns contained string data and which contained numeric data by analyzing the presence of NaN values in the temporary_mean array. I then proceeded to load the string data from the dataset into loan_data_strings, while ensuring proper formatting by replacing commas with dots for decimal points through the comma_to_dot function.
```python
column_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()
column_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze() 
loan_data_strings = np.genfromtxt("loan-data.csv", 
                                  delimiter = ';', 
                                  skip_header = 1, 
                                  autostrip = True,
                                  usecols = column_strings,
                                  dtype = str)
def comma_to_dot(x):
    return float(x.decode('utf-8').replace(',', '.'))
loan_data_numeric = np.genfromtxt("loan-data.csv", 
                                  delimiter=';', 
                                  skip_header=1, 
                                  autostrip=True,
                                  usecols=column_numeric,
                                  filling_values=temporary_fill,
                                  converters={col: comma_to_dot for col in column_numeric})
```
To ensure the dataset was well-structured, I extracted the full header information from the CSV file by reading only the header row. From there, I categorized the headers into two groups: string columns and numeric columns.
```python
header_full = np.genfromtxt("loan-data.csv", 
                                  delimiter = ';', 
                                  autostrip = True,
                                  skip_footer = raw_data_np.shape[0],
                                  dtype = str)
header_strings, header_numeric = header_full[column_strings], header_full[column_numeric]
```

To maintain organized backups throughout the data processing workflow, I created a custom function checkpoint. This method provided a reliable way to store interim data states and seamlessly resume processing at any point.
```python
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable
```
### String to Numeric Conversion
In this step, I updated the first header to "issue_date" and examined the unique values in the relevant data column. 
Ater it I cleaned up the column by removing any extraneous text and mapped month abbreviations to their numerical equivalents for easier future work with data set.
```python
header_strings[0] = 'issue_date'
np.unique(loan_data_strings[:,0])
loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], '-15')
months = np.array(['', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'])
for i in range(13):
    loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i],
                                     i,
                                     loan_data_strings[:,0]) np.unique(loan_data_strings[:,0])
```

I analyzed the unique values in the loan status column and categorized certain statuses like "Charged Off" and "Default" as 0, while other statuses were set to 1. This binary classification simplifies the dataset for more straightforward processing.
```python
np.unique(loan_data_strings[:,1])
status_bad = np.array(['', 'Charged Off', 'Default','Late (31-120 days)'])
loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad), 0, 1)
```








