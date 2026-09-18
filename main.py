"""
Medical Insurance Data Analysis

Author: Neha Vutukuri

Project Description:
This project uses Python, Pandas, and Matplotlib to analyze a real-world
medical insurance dataset. The goal is to answer questions about insurance
charges by exploring relationships between age, BMI, smoking status,
region, sex, and number of children using descriptive statistics,
grouped analysis, and data visualizations.
"""

#===========================================================#
# IMPORT LIBRARIES
#===========================================================#


import pandas as pd
import matplotlib.pyplot as plt


#===========================================================#
# LOAD DATASET
#===========================================================#


# Loads the medical insurance dataset into a DataFrame called df
df = pd.read_csv("insurance.csv")


#===========================================================#
# DATA EXPLORATION
#===========================================================#


# Shows the first 5 rows to understand what the dataset looks like
print("First 5 rows of the dataset:")
print(df.head())

# Counts how many people (rows) are in the dataset
print()
print("Number of people in the dataset:")
print(len(df))

# Displays the names of all the categories/information available in the dataset
print()
print("Column names:")
print(df.columns)

# Counts how many different pieces of information (columns) are in the dataset
print()
print("Number of columns:")
print(len(df.columns))


#===========================================================#
# BASIC STATISTICS
#===========================================================#


# Provides a statistical summary of the numerical data in the dataset
# Includes values like count, average, minimum, maximum, and standard deviation
print()
print("Basic statistics summary:")
print(df.describe())

# Calculates the average age of everyone in the dataset
print()
print("Average age:")
print(df["age"].mean())

# Finds the oldest person in the dataset
print()
print("Oldest person:")
print(df["age"].max())

# Finds the youngest person in the dataset
print()
print("Youngest person:")
print(df["age"].min())

# Calculates the average BMI of everyone in the dataset
print()
print("Average BMI:")
print(f"{df['bmi'].mean():.2f}")

# Calculates the average insurance cost for everyone in the dataset
print()
print("Average insurance charge:")
print(f"${df['charges'].mean():,.2f}")

# Finds the highest insurance charge in the dataset
print()
print("Highest insurance charge:")
print(f"${df['charges'].max():,.2f}")

# Finds the lowest insurance charge in the dataset
print()
print("Lowest insurance charge:")
print(f"${df['charges'].min():,.2f}")


#===========================================================#
# SMOKING ANALYSIS
#===========================================================#


# Filter the dataset to only include smokers, then calculate their average insurance cost
print()
print("Average insurance charge for smokers:")
print(f"${df[df['smoker'] == 'yes']['charges'].mean():,.2f}")

# Filter the dataset to only include non-smokers, then calculate their average insurance cost
print()
print("Average insurance charge for non-smokers:")
print(f"${df[df['smoker'] == 'no']['charges'].mean():,.2f}")

# Counts the number of smokers in the dataset
print()
print("Number of smokers:")
print(len(df[df["smoker"] == "yes"]))

# Counts the number of non-smokers in the dataset
print()
print("Number of non-smokers:")
print(len(df[df["smoker"] == "no"]))

# Calculates the percentage of people who are smokers
print()
print("Percentage of people who DO smoke:")
print((len(df[df["smoker"] == "yes"]) / len(df)) * 100, "%")

# Calculates the percentage of people who are non-smokers
print()
print("Percentage of people who DO NOT smoke:")
print((len(df[df["smoker"] == "no"]) / len(df)) * 100, "%")


#===========================================================#
# REGIONAL ANALYSIS
#===========================================================#


# Finds all unique regions in the dataset
print()
print("Regions included in the dataset:")
print(df["region"].unique())

# Calculates the average insurance charge for each region
print()
print("Average insurance charge by region:")
print(df.groupby("region")["charges"].mean().map(lambda x: f"${x:,.2f}"))

# Counts how many people are in each region
print()
print("Number of people in each region:")
print(df["region"].value_counts())

# Counts the number of smokers in each region
print()
print("Number of smokers by region:")
print(df[df["smoker"] == "yes"]["region"].value_counts())

# Counts smokers and non-smokers in each region:
print()
print("Proportion of smokers and non-smokers by region:")
print(df.groupby("region")["smoker"].value_counts())

# Calculates the percentage of smokers and non-smokers within each region
print()
print("Percentage of smokers and non-smokers by region:")
print(df.groupby("region")["smoker"].value_counts(normalize=True) * 100)

# Calculates the average BMI for each region
print()
print("Average BMI by region:")
print(df.groupby("region")["bmi"].mean().map(lambda x: f"{x:.2f}"))


#===========================================================#
# AGE ANALYSIS
#===========================================================#


# Displays the 10 oldest people
print()
print("10 oldest people:")
print(df.sort_values("age", ascending=False).head(10))

# Displays the 10 youngest people
print()
print("10 youngest people:")
print(df.sort_values("age").head(10))

# Creates age groups for easier analysis
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 40, 50, 65, 100],
    labels=["18-30", "31-40", "41-50", "51-65", "66+"],
    include_lowest=True
)

# Counts people in each age group
print()
print("Number of people in each age group:")
print(df["age_group"].value_counts().sort_index())

# Counts smokers in each age group
print()
print("Number of smokers in each age group:")
print(df[df["smoker"] == "yes"]["age_group"].value_counts().sort_index())

# Displays percentage of smokers by age group
print()
print("Percentage of smokers by age group:")
print(df.groupby("age_group")["smoker"].value_counts(normalize=True) * 100)


#===========================================================#
# CHILDREN ANALYSIS
#===========================================================#


# Displays all unique values in the children column
print()
print("Possible number of children:")
print(df["children"].unique())

# Counts how many people have each number of children
print()
print("Number of people by number of children:")
print(df["children"].value_counts().sort_index())

# Calculates the average insurance charge based on the number of children
print()
print("Average insurance charge by number of children:")
print(df.groupby("children")["charges"].mean().map(lambda x: f"${x:,.2f}"))


#===========================================================#
# SEX ANALYSIS
#===========================================================#


# Counts the number of males and females
print()
print("Number of people by sex:")
print(df["sex"].value_counts())

# Calculates the average insurance charge by sex
print()
print("Average insurance charge by sex:")
print(df.groupby("sex")["charges"].mean().map(lambda x: f"${x:,.2f}"))


#===========================================================#
# BMI ANALYSIS
#===========================================================#


# Creates BMI categories based on commonly used BMI ranges
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

# Calculates the average insurance charge by BMI category
average_bmi_category = df.groupby("bmi_category")["charges"].mean()

print()
print("Average insurance charge by BMI category:")
print(average_bmi_category.map(lambda x: f"${x:,.2f}"))


#===========================================================#
# HIGHEST AND LOWEST INSURANCE CHARGES
#===========================================================#


# Sorts the dataset by insurance charges (highest to lowest)
print()
print("10 most expensive customers:")
print(df.sort_values("charges", ascending=False).head(10))

# Sorts the dataset by insurance charges (lowest to highest)
print()
print("10 least expensive customers:")
print(df.sort_values("charges").head(10))


#===========================================================#
# DATA VISUALIZATION
#===========================================================#


# Creates a scatter plot of age vs. insurance charges
plt.scatter(df["age"], df["charges"], s=20, alpha=0.7)

plt.title("Age vs. Insurance Charges")

plt.xlabel("Age")

plt.ylabel("Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/age_vs_charges.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Calculates average insurance charges for each region
region_average = df.groupby("region")["charges"].mean()

# Creates a bar chart comparing average insurance charges by region
plt.figure()

plt.bar(
    region_average.index,
    region_average.values,
)

plt.title("Average Insurance Charges by Region")

plt.xlabel("Region")

plt.ylabel("Average Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/region_charges.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Creates a bar chart comparing average insurance charges for each BMI category
plt.figure()

plt.bar(
    average_bmi_category.index.astype(str),
    average_bmi_category.values
)

plt.title("Average Insurance Charges by BMI Category")

plt.xlabel("BMI Category")

plt.ylabel("Average Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/charges_by_bmi.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Creates a scatter plot of BMI vs. insurance charges
plt.figure()

plt.scatter(df["bmi"], df["charges"], s=20, alpha=0.7)

plt.title("BMI vs. Insurance Charges")

plt.xlabel("BMI")

plt.ylabel("Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/bmi_vs_charges.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Calculates the average insurance charge for smokers(saved as a variable)
smoker_average = df[df["smoker"] == "yes"]["charges"].mean()

# Calculates the average insurance charge for non-smokers(saved as a variable)
non_smoker_average = df[df["smoker"] == "no"]["charges"].mean()

# Creates a bar chart comparing average insurance charges for smokers and non-smokers
plt.figure()

plt.bar(
    ["Smokers", "Non-Smokers"],
    [smoker_average, non_smoker_average],
)

plt.title("Average Insurance Charges: Smokers vs. Non-Smokers")

plt.xlabel("Smoking Status")

plt.ylabel("Average Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/smoker_vs_nonsmoker.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Creates a pie chart showing smokers vs. non-smokers
plt.figure()

plt.pie(
    [len(df[df["smoker"] == "yes"]),
        len(df[df["smoker"] == "no"]),
    ],
    labels=["Smokers", "Non-Smokers"],
    autopct="%1.1f%%"
)

plt.title("Smoking Status in the Dataset")

plt.tight_layout()

plt.savefig("visualizations/smoking_status.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Creates a bar chart comparing average insurance charges by sex
plt.figure()

average_by_sex = df.groupby("sex")["charges"].mean()

plt.bar(
    average_by_sex.index,
    average_by_sex.values
)

plt.title("Average Insurance Charges by Sex")

plt.xlabel("Sex")

plt.ylabel("Average Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/charges_by_sex.png", dpi=300)

plt.show()

#-----------------------------------------------------------#

# Creates a bar chart showing average insurance charges by number of children
plt.figure()

average_by_children = df.groupby("children")["charges"].mean()

plt.bar(
    average_by_children.index,
    average_by_children.values
)

plt.title("Average Insurance Charges by Number of Children")

plt.xlabel("Number of Children")

plt.ylabel("Average Insurance Charges ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/charges_by_children.png", dpi=300)

plt.show()


#===========================================================#
# PROJECT FINDINGS
#===========================================================#


# 1. Smokers had substantially higher average insurance charges
#    than non-smokers.

# 2. Insurance charges generally increased with age, although
#    age alone did not explain all differences in insurance costs.

# 3. The Southeast region had the highest average insurance
#    charges among the four regions.

# 4. The Southeast region also had the highest percentage of smokers,
#    which may contribute to the regional difference in average charges.

# 5. Average insurance charges varied across BMI categories,
#    although these differences were smaller than the difference
#    between smokers and non-smokers.

# 6. Many customers with the highest insurance charges were
#    older or smokers, although individual costs varied widely.

# 7. The number of children showed relatively small differences
#    in average charges compared with factors such as smoking
#    status and age.


#===========================================================#
# PROJECT CONCLUSION
#===========================================================#

"""
This project explored a real-world medical insurance dataset using
Python, Pandas, and Matplotlib.

By applying filtering, grouping, sorting, descriptive statistics,
and data visualization techniques I answered questions about how
different factors relate to insurance charges.

Smoking status showed the largest difference in average insurance 
charges, while age showed a general positive relationship with 
insurance charges. The other factors I analyzed, including 
BMI, region, sex, and number of children, showed smaller differences 
in average insurance charges.

Overall, this project helped me strengthen my ability to work with
real-world data, use Python to answer questions, and communicate
findings through data visualizations.
"""
