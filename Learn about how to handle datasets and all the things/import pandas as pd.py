import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load open source data from the web (replace with actual URL)
file_path = "C:\\Users\\harsh\\Downloads\\House Price.csv"  # Corrected for Windows file paths
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the file path and try again.")

# Display first 5 rows
print(data.head().to_markdown(index=False,numalign="left", stralign="left"))

# Display last 5 rows
print(data.tail().to_markdown(index=False,numalign="left", stralign="left"))

# Column names
print(data.columns.tolist())

# Count of non-null values in each column
print(data.count().to_markdown(numalign="left", stralign="left"))

# Data types and non-null values
print(data.info())

# Summary statistics
print(data.describe().to_markdown())

# Filter relevant columns
data = data[["Area","BHK","Bathroom","Parking","Price","Per_Sqft"]]

# Correlation matrix
correlation = data.corr()
print(correlation.to_markdown(numalign="left", stralign="left"))

# Correlation with 'Price' (in percentage)
print((correlation["Price"] * 100).to_markdown(numalign="left", stralign="left"))

# Calculate and print max, min, and range of 'Price'
mx = data['Price'].max()
mn = data['Price'].min()
print(f"max:{mx}\tmin:{mn}\nrange:{mx-mn}")

# Standard deviation of 'Price'
print(data["Price"].std())

# Check for missing values
print(data.isnull().sum().to_markdown(numalign="left", stralign="left"))

# Fill missing values in 'Per_Sqft' with mean
data['Per_Sqft'] = data['Per_Sqft'].fillna(data['Per_Sqft'].mean())

# Correlation between 'Per_Sqft' and 'Price'
print((data[['Per_Sqft','Price']].corr() * 100).to_markdown(numalign="left", stralign="left"))

# Check for missing values
print(data.isnull().sum().to_markdown(numalign="left", stralign="left"))

# Types of BHK and their count
types = np.sort(data['BHK'].unique())
for i in types:
    print(f"BHK: {i}")
    print(data[data['BHK'] == i].count().to_markdown(numalign="left", stralign="left"))
    print()

# Interquartile range (IQR) for each column
for i in data.columns:
    print(f"IQR of {i}: {np.percentile(data[i],75) - np.percentile(data[i],25)}")

# Check for skewness in each column
for i in data.columns:
    mean = data[i].mean()
    median = data[i].median()
    if mean < median:
        print(f"{i} is left skewed (mean: {mean}, median: {median})")
    else:
        print(f"{i} is right skewed (mean: {mean}, median: {median})")

# Maximum and minimum values for each column
for i in data.columns:
    print(f"Max_{i}: {np.max(data[i])}\tMin_{i}: {np.min(data[i])}")

# Normalization (scaling) of each column
for i in data.columns:
    data[i] = data[i] / data[i].max()

print(data.describe().to_markdown(numalign="left", stralign="left"))

# Statistical analysis of 'Area'
df = data
print(f"Sum of Area: {df['Area'].sum()}")
print(f"Max Area: {df['Area'].max()}")
print(f"Min Area: {df['Area'].min()}")
print(f"Mode of Area: {df['Area'].mode().tolist()}")
print(f"Median Area: {df['Area'].median()}")
print(f"Standard Deviation of Area: {df['Area'].std()}")
print(f"50th Percentile of Area: {np.percentile(df['Area'],50)}")

import statistics as st
from scipy.stats import skew, kurtosis
import scipy

print(f"Harmonic Mean of Area: {st.harmonic_mean(df['Area'])}")
print(f"Skewness of Area: {skew(df['Area'])}")
print(f"Kurtosis of Area: {kurtosis(df['Area'])}")

print(f"Median Grouped (1) of Area: {st.median_grouped(df['Area'],1)}")
print(f"Median Grouped (2) of Area: {st.median_grouped(df['Area'],2)}")

print(f"Variance of Area: {st.variance(df['Area'])}")
print(f"Standard Deviation of Area: {st.stdev(df['Area'])}")
print(f"Geometric Mean of Area: {scipy.stats.gmean(df['Area'])}")
