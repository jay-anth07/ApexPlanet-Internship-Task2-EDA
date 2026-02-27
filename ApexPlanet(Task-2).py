import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)        # Show all rows
pd.set_option('display.max_columns', None)     # Show all columns
pd.set_option('display.width', None)           # Auto-detect width
pd.set_option('display.max_colwidth', None)    # Show full column content

# Load dataset
df = pd.read_csv("cleaned_superstore.csv")  # This Dataset is output of the first Task

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# Column names
print("\nColumns in Dataset:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# After converting the dataset into csv in the first task the dates are converted into objects .
#So I'm going Convert Order Date and Ship Date back to datetime
# Convert Order Date and Ship Date automatically
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Verify
print(df[['Order Date', 'Ship Date']].dtypes) #Converted .......

#analyze numerical columns only:,Sales,Shipping Days,Postal Code (just structurally, not business-wise),Order Year
# Select numerical columns
numerical_cols = ['Sales', 'Shipping Days', 'Postal Code', 'Order Year']

print("Descriptive Statistics:\n")
print(df[numerical_cols].describe().T)

# Graphs
plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df['Sales']), bins=50, kde=True)
plt.title("Log-Transformed Sales Distribution")
plt.xlabel("Log(Sales)")
plt.ylabel("Frequency")
plt.show()
print("Outlier Detection:")
#Outelier Analysis

# Create log sales column
df['Log_Sales'] = np.log1p(df['Sales'])

# Calculate Q1 and Q3
Q1 = df['Log_Sales'].quantile(0.25)
Q3 = df['Log_Sales'].quantile(0.75)

IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df['Log_Sales'] < lower_bound) | (df['Log_Sales'] > upper_bound)]

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Outliers:", len(outliers))

print("\nCategorical Analysis")
category_analysis = df.groupby('Category')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
print(category_analysis)
print()
print("Sub-Category Analysis ")
sub_category_analysis = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).reset_index()
print(sub_category_analysis.head(10))
print()
print("Region-wise Sales Performance")
region_analysis = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
print(region_analysis)
print()
print("Category × Region Analysis")
category_region = df.pivot_table(values='Sales',
                                 index='Region',
                                 columns='Category',
                                 aggfunc='sum')
print(category_region)
print()
print("Time-Based Analysis (Trend Analysis)")
yearly_sales = df.groupby('Order Year')['Sales'].sum().reset_index()
print(yearly_sales)
print()
print("Monthly Trend Analysis")
monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()
print(monthly_sales.sort_values(by='Sales', ascending=False))
print()
print("Sales Distribution Visualization")

plt.hist(df['Sales'], bins=50)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()
print()

segment_analysis = df.groupby('Segment')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
print("Segment Analysis: \n",segment_analysis)
