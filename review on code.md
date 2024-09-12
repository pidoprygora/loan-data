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

In the next step, I identified which columns contained string data and which contained numeric data by analyzing the presence of NaN values in the temporary_mean array. And then I devided data into separate string and numeric arrays.
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
Afterward, I applied the same categorization to the headers.
```python
header_full = np.genfromtxt("loan-data.csv", 
                                  delimiter = ';', 
                                  autostrip = True,
                                  skip_footer = raw_data_np.shape[0],
                                  dtype = str)
header_strings, header_numeric = header_full[column_strings], header_full[column_numeric]
```

To keep backups organized during data processing, I created a custom function called checkpoint. This made it easy to save and restore data at any stage.
```python
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable
```
### String to Numeric Conversion
#### >issue_d column
##### Before
```python
['', 'Apr-15', 'Aug-15', 'Dec-15', 'Feb-15', 'Jan-15', 'Jul-15',
       'Jun-15', 'Mar-15', 'May-15', 'Nov-15', 'Oct-15', 'Sep-15']
```
In this step, I began by renaming the header and removing unnecessary parts of the data. Then, I converted the month names to their numeric equivalents.
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
##### After
```python
['0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9']
```

#### >loan_status column
##### Befor
```python
['', 'Charged Off', 'Current', 'Default', 'Fully Paid',
       'In Grace Period', 'Issued', 'Late (16-30 days)',
       'Late (31-120 days)']
```
Having such a data, I divided it into two groups: 1 for positive statuses and 0 for negative and empty statuses.
```python
np.unique(loan_data_strings[:,1])
status_bad = np.array(['', 'Charged Off', 'Default','Late (31-120 days)'])
loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad), 0, 1)
```
##### After
```python
['0', '1']
```

#### >term column
##### Before
```python
['', '36 months', '60 months']
```
I cleaned the data in the third column by removing the text " months" and renamed the corresponding header to "term_months." Empty values were then replaced with '60'  as this can be the worst-case scenario.
```python
loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], ' months')
header_strings[2] = 'term_months'
loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '', '60', loan_data_strings[:,2])
```
##### After 
```python
['36', '60']
```

#### >grade and sub_grade
##### Before
```python
['', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
['', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
       'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1',
       'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2',
       'G3', 'G4', 'G5']
```
In this step, I filled missing values in the fifth column based on the data in the fourth column, appending '5' to non-empty entries. Any remaining empty values were set to 'H1' as the worst possible case. I also deleted the fourth column and its corresponding header to streamline the dataset.
```python
for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '')&(loan_data_strings[:,3] == i),
                                      i + '5',
                                      loan_data_strings[:,4])
loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == ''), 'H1', loan_data_strings[:,4])
loan_data_strings = np.delete(loan_data_strings, 3, axis = 1)
header_strings = np.delete(header_strings, 3)
```
I created a dictionary to map unique values from the fourth column to numerical codes and used this mapping to convert the categorical values into corresponding numeric codes.
```python
keys = list(np.unique(loan_data_strings[:,3]))
values = list(range(1, len(np.unique(loan_data_strings[:,3])) + 1))
dict_sub_grade = dict(zip(keys, values))
for i in np.unique(loan_data_strings[:,3]):
    loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i,
                                      dict_sub_grade[i],
                                      loan_data_strings[:,3])
```
##### After 
```python
['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
       '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
       '3', '30', '31', '32', '33', '34', '35', '36', '4', '5', '6', '7',
       '8', '9']
```

#### >verification_status
##### Before
```python
['', 'Not Verified', 'Source Verified', 'Verified']
```
With this data, I also categorized it into two groups: 1 for positive statuses and 0 for negative or empty statuses.
```python
loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), 0, 1)
```
##### After 
```python
['0', '1']
```

#### >addr_state
##### Before
```python
['CA', 'NY', 'TX', 'FL', '', 'IL', 'NJ', 'GA', 'PA', 'OH', 'MI',
        'NC', 'VA', 'MD', 'AZ', 'WA', 'MA', 'CO', 'MO', 'MN', 'IN', 'WI',
        'CT', 'TN', 'NV', 'AL', 'LA', 'OR', 'SC', 'KY', 'KS', 'OK', 'UT',
        'AR', 'MS', 'NH', 'NM', 'WV', 'HI', 'RI', 'MT', 'DE', 'DC', 'WY',
        'AK', 'NE', 'SD', 'VT', 'ND', 'ME']
```
I first replaced any empty values in the fifth column with 0. Then, I categorized the states into four regions—West, South, Midwest, and East—assigning numeric codes (1 through 4) to each region based on their state abbreviations.
```python
loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '', 0, loan_data_strings[:,5])  states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])

states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])

loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5],states_west ), 1, loan_data_strings[:,5]) 
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5],states_south), 2, loan_data_strings[:,5]) 
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5],states_midwest ), 3, loan_data_strings[:,5]) 
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5],states_east ), 4, loan_data_strings[:,5]) 
```
##### After 
```python
['0', '1', '2', '3', '4']
```
#### >final step
I converted the data to integers and saved a checkpoint with the changed headers and data.
```python
loan_data_strings = loan_data_strings.astype(int)
checkpoint_strings = checkpoint("checkpoint-string", header_strings, loan_data_strings)
```

### Cleaning Numeric data
To address missing data, I categorized the columns into two groups: one where missing values are filled with the minimum value and another where they are filled with the maximum value.
#### >funded_amnt
I replaced missing values in the numeric data by substituting placeholders with the minimum value for that column.
```python
loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temporary_fill, 
                                  temporary_stats[0, column_numeric[2]],
                                  loan_data_numeric[:,2])
```

#### >other numeric column
I updated specific columns in the numeric data by replacing placeholders with the maximum value for each column, based on a predefined list of column indices.
```python
for i in [1, 3, 4, 5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporary_fill, 
                                  temporary_stats[0, column_numeric[i]],
                                  loan_data_numeric[:,i])
```

### Adding EUR currency
I first loaded the EUR to USD exchange rates from a CSV file and then matched these rates with the corresponding months in the exchange_rate array. Missing exchange rates were filled with the average rate.
```python
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
EUR_USD exchange_rate = loan_data_strings[:,0]
```
#### >creating EUR colums
Next, I adjusted the numeric data for columns with dollar values by converting them to EUR using the updated exchange rates.
```python
for i in range(1, 13):
    exchange_rate = np.where(exchange_rate == i,
                            EUR_USD[i-1],
                            exchange_rate)
    
exchange_rate = np.where(exchange_rate == 0,
                            np.mean(EUR_USD),
                            exchange_rate)
```
I appended these new EUR columns to the existing numeric data and updated the headers to reflect the new columns.
```python
exchange_rate for i in colums_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:,i]/loan_data_numeric[:,6],(10000, 1))))
header_aditional = np.array([column_name + '_EUR' for column_name in header_numeric[colums_dollar]])
header_numeric = np.concatenate((header_numeric, header_aditional))
```

#### >reordering columns and headers
##### Before
```python
['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment',
       'total_pymnt', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR',
       'installment_EUR', 'total_pymnt_EUR']
```
I reordered the columns in the numeric data and headers to ensure that costs in different currencies are displayed next to each other.
```python
column_index_order = [0, 1, 7, 2, 8, 3, 4, 9, 5, 10, 6]
header_numeric = header_numeric[column_index_order] l
loan_data_numeric = loan_data_numeric[:,column_index_order]
```
##### After 
```python
['id', 'loan_amnt', 'loan_amnt_EUR', 'funded_amnt',
       'funded_amnt_EUR', 'int_rate', 'installment', 'installment_EUR',
       'total_pymnt', 'total_pymnt_EUR', 'exchange_rate']
```

#### >final step
I saved a checkpoint for the numeric data and headers to allow for easy retrieval and continuation of work.
```python
checkpoint_numeric = checkpoint("checkpoint-numeric", header_numeric, loan_data_numeric)
```

### Saving new dataset
I saved the preprocessed loan data to a CSV file named loan-data-preprocessed.csv using a comma as the delimiter and specifying the format for each value as a string.
```python
np.savetxt('loan-data-preprocessed.csv',
           loan_data,
           fmt="%s",
           delimiter=',')
```

## Conclusion
In summary, I completed the preprocessing of the loan data by cleaning and transforming various fields, applying necessary conversions, and reordering columns for better organization. The final dataset was saved in a CSV file, ready for further analysis or reporting.










