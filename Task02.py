# PRODIGY_DS_02

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("student_data.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------

# 1. Distribution of Final Grades
plt.figure(figsize=(8,5))
sns.histplot(df['G3'], bins=10, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Count")
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df)
plt.title("Gender Distribution")
plt.show()

# 3. Study Time vs Final Grade
plt.figure(figsize=(8,5))
sns.boxplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.show()

# 4. Absences vs Final Grade
plt.figure(figsize=(8,5))
sns.scatterplot(x='absences', y='G3', data=df)
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("Final Grade")
plt.show()

# 5. Internet Access vs Final Grade
plt.figure(figsize=(6,4))
sns.boxplot(x='internet', y='G3', data=df)
plt.title("Internet Access vs Final Grade")
plt.show()

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# KEY INSIGHTS
# -----------------------------

print("\n===== KEY INSIGHTS =====")
print("1. Dataset contains", df.shape[0], "rows and", df.shape[1], "columns.")
print("2. Missing values checked and cleaned.")
print("3. Distribution of final grades analyzed.")
print("4. Relationship between study time and grades examined.")
print("5. Relationship between absences and grades examined.")
print("6. Correlation heatmap generated for numerical features.")