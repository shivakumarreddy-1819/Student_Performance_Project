import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/StudentsPerformance.csv")

print("\nSTUDENT PERFORMANCE CASE STUDY")
print("=" * 40)

print("Number of students:", len(data))
print("Number of columns:", len(data.columns))


# ==========================================
# 2. MISSING VALUE ANALYSIS
# ==========================================

print("\nMISSING VALUE ANALYSIS")
print("=" * 40)

missing_values = data.isnull().sum()

print(missing_values)

print("\nTotal missing values:", data.isnull().sum().sum())


# ==========================================
# 3. DUPLICATE ANALYSIS
# ==========================================

print("\nDUPLICATE ANALYSIS")
print("=" * 40)

duplicates = data.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# ==========================================
# 4. DATA TYPES
# ==========================================

print("\nDATA TYPES")
print("=" * 40)

print(data.dtypes)


# ==========================================
# 5. BASIC STATISTICAL SUMMARY
# ==========================================

print("\nSTATISTICAL SUMMARY")
print("=" * 40)

print(data.describe())


# ==========================================
# 6. MATH SCORE ANALYSIS
# ==========================================

math_score = data["Math score"]

print("\nMATH SCORE ANALYSIS")
print("=" * 40)

print("Mean:", math_score.mean())
print("Median:", math_score.median())
print("Variance:", math_score.var())
print("Standard Deviation:", math_score.std())
print("Minimum:", math_score.min())
print("Maximum:", math_score.max())
# ==========================================
# 7. NUMERICAL FEATURE IMPACT ANALYSIS
# ==========================================

print("\nNUMERICAL FEATURE IMPACT ANALYSIS")
print("=" * 40)

# Pearson correlation between Reading Score and Math Score
reading_corr = data["Reading score"].corr(data["Math score"])

# Pearson correlation between Writing Score and Math Score
writing_corr = data["Writing score"].corr(data["Math score"])

print("Reading Score vs Math Score")
print("Pearson correlation:", round(reading_corr, 3))

print("\nWriting Score vs Math Score")
print("Pearson correlation:", round(writing_corr, 3))
# ==========================================
# 8. CATEGORICAL FEATURE IMPACT ANALYSIS
# ==========================================

print("\nCATEGORICAL FEATURE IMPACT ANALYSIS")
print("=" * 40)

# Gender
print("\nGender - Average Math Score")
print(data.groupby("Gender")["Math score"].mean().round(2))

# Race/Ethnicity
print("\nRace/Ethnicity - Average Math Score")
print(data.groupby("race/ethnicity")["Math score"].mean().round(2))

# Parental Level of Education
print("\nParental Level of Education - Average Math Score")
print(
    data.groupby("Parental level of education")["Math score"]
    .mean()
    .round(2)
)

# Lunch
print("\nLunch - Average Math Score")
print(data.groupby("Lunch")["Math score"].mean().round(2))

# Test Preparation Course
print("\nTest Preparation Course - Average Math Score")
print(
    data.groupby("Test preparation course")["Math score"]
    .mean()
    .round(2)
)
# ==========================================
# 9. DATA VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt
import os

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)


# ------------------------------------------
# Graph 1: Math Score Distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data["Math score"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("graphs/math_score_distribution.png")
plt.close()


# ------------------------------------------
# Graph 2: Reading Score vs Math Score
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Reading score"],
    data["Math score"],
    alpha=0.5
)

plt.title("Reading Score vs Math Score")
plt.xlabel("Reading Score")
plt.ylabel("Math Score")

plt.tight_layout()
plt.savefig("graphs/reading_vs_math.png")
plt.close()


# ------------------------------------------
# Graph 3: Writing Score vs Math Score
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Writing score"],
    data["Math score"],
    alpha=0.5
)

plt.title("Writing Score vs Math Score")
plt.xlabel("Writing Score")
plt.ylabel("Math Score")

plt.tight_layout()
plt.savefig("graphs/writing_vs_math.png")
plt.close()


# ------------------------------------------
# Graph 4: Average Math Score by Gender
# ------------------------------------------

gender_mean = data.groupby("Gender")["Math score"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    gender_mean.index,
    gender_mean.values,
    edgecolor="black"
)

plt.title("Average Math Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Math Score")

plt.tight_layout()
plt.savefig("graphs/gender_comparison.png")
plt.close()


print("\nDATA VISUALIZATION")
print("=" * 40)
print("Graphs generated successfully.")
print("Graphs are saved in the 'graphs' folder.")
# ==========================================
# 10. PROBABILITY DISTRIBUTION ANALYSIS
# ==========================================

print("\nPROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 40)

# Total number of students
total_students = len(data)

# Students scoring between 50 and 70
students_50_70 = data[
    (data["Math score"] >= 50) &
    (data["Math score"] <= 70)
]

count_50_70 = len(students_50_70)

# Experimental probability
probability_50_70 = count_50_70 / total_students

print("Students scoring between 50 and 70:", count_50_70)
print("Total students:", total_students)

print(
    "P(50 <= Math Score <= 70):",
    round(probability_50_70, 3)
)

print(
    "Probability percentage:",
    round(probability_50_70 * 100, 2),
    "%"
)
# ==========================================
# 11. NORMAL DISTRIBUTION ANALYSIS
# ==========================================

from scipy.stats import norm

print("\nNORMAL DISTRIBUTION ANALYSIS")
print("=" * 40)

# Mean and standard deviation
mean = data["Math score"].mean()
std = data["Math score"].std()

# Z-scores
z_50 = (50 - mean) / std
z_70 = (70 - mean) / std

# Normal distribution probability
normal_probability = norm.cdf(z_70) - norm.cdf(z_50)

print("Mean (μ):", round(mean, 3))
print("Standard Deviation (σ):", round(std, 3))

print("Z-score for 50:", round(z_50, 3))
print("Z-score for 70:", round(z_70, 3))

print(
    "Normal Distribution Probability:",
    round(normal_probability, 4)
)

print(
    "Normal Distribution Percentage:",
    round(normal_probability * 100, 2),
    "%"
)
# ==========================================
# 12. BINOMIAL DISTRIBUTION ANALYSIS
# ==========================================

from math import comb

print("\nBINOMIAL DISTRIBUTION ANALYSIS")
print("=" * 40)

# Assumption: Math Score >= 40 is considered a pass
pass_count = (data["Math score"] >= 40).sum()
fail_count = (data["Math score"] < 40).sum()

total_students = len(data)

p = pass_count / total_students
q = 1 - p

print("Pass students:", pass_count)
print("Fail students:", fail_count)

print("Probability of passing (p):", round(p, 3))
print("Probability of failing (q):", round(q, 3))

# Number of students selected
n = 5

# Probability of exactly 4 passing
k = 4

probability_4 = (
    comb(n, k) *
    (p ** k) *
    (q ** (n - k))
)

# Probability of exactly 5 passing
k = 5

probability_5 = (
    comb(n, k) *
    (p ** k) *
    (q ** (n - k))
)

print("\nFor", n, "randomly selected students:")

print(
    "P(X = 4):",
    round(probability_4, 4),
    "(",
    round(probability_4 * 100, 2),
    "%)"
)

print(
    "P(X = 5):",
    round(probability_5, 4),
    "(",
    round(probability_5 * 100, 2),
    "%)"
)
# ==========================================
# 13. REGRESSION AND LEAST SQUARES
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\nREGRESSION AND LEAST SQUARES")
print("=" * 40)

# Input features
X = data[["Reading score", "Writing score"]]

# Target variable
y = data["Math score"]

# Create linear regression model
model = LinearRegression()

# Train the model using the complete dataset
model.fit(X, y)

# Generate predictions
predictions = model.predict(X)

# Regression coefficients
intercept = model.intercept_
reading_coefficient = model.coef_[0]
writing_coefficient = model.coef_[1]

print("Intercept:", round(intercept, 3))
print("Reading coefficient:", round(reading_coefficient, 3))
print("Writing coefficient:", round(writing_coefficient, 3))

print("\nRegression Equation:")
print(
    "Math Score =",
    round(intercept, 3),
    "+",
    round(reading_coefficient, 3),
    "× Reading Score +",
    round(writing_coefficient, 3),
    "× Writing Score"
)

# Model evaluation
r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

print("\nModel Evaluation")
print("R² Score:", round(r2, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
# ==========================================
# 14. SAMPLE STUDENT PREDICTION
# ==========================================

print("\nSAMPLE STUDENT PREDICTION")
print("=" * 40)

# Example student scores
sample_reading = 75
sample_writing = 70

# Create input with the same feature names used during training
sample_student = pd.DataFrame({
    "Reading score": [sample_reading],
    "Writing score": [sample_writing]
})

# Predict Math Score
sample_prediction = model.predict(sample_student)[0]

print("Reading Score:", sample_reading)
print("Writing Score:", sample_writing)
print("Predicted Math Score:", round(sample_prediction, 2))