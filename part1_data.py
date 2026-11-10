import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n = 150

study_hours = np.concatenate([
    np.random.uniform(0.5, 2.5, 50),
    np.random.uniform(2.5, 5.0, 50),
    np.random.uniform(5.0, 8.0, 50),
])

attendance = np.concatenate([
    np.random.uniform(40, 65, 50),
    np.random.uniform(65, 80, 50),
    np.random.uniform(80, 100, 50),
])

exam_score = np.concatenate([
    np.random.uniform(30, 55, 50),
    np.random.uniform(55, 75, 50),
    np.random.uniform(75, 100, 50),
])

df = pd.DataFrame({
    "study_hours_per_day": np.round(study_hours, 2),
    "attendance_percentage": np.round(attendance, 2),
    "exam_score": np.round(exam_score, 2),
})

print("Raw data preview:")
print(df.head(10))
print()

print("Any missing values?", df.isnull().sum().sum())
print("Total rows:", len(df))
print()

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_values, columns=df.columns)

output_path = "Tadena_student_data.csv"
df_scaled.to_csv(output_path, index=False)

print("Cleaned and standardized data saved to:", output_path)
print()
print("Scaled data preview:")
print(df_scaled.head(10))
