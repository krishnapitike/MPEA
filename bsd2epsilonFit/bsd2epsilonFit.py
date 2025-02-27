import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load data from the file
file_path = "../data/Borges.txt"  # Replace with your actual file path
df = pd.read_csv(file_path, sep="\t")

# Filter the rows where gocc/gbond is between 0.77 and 0.975
df_gocc = df[(df["gocc/gbond"] >= 0.8) & (df["gocc/gbond"] <= 0.975)]

# Define the features and target for gocc/gbond
X_gocc = df_gocc["gocc/gbond"].values.reshape(-1, 1)
y_gocc = df_gocc["epsilon_p [%]"].values

# Create a range of x values for fitting between min and max of gocc/gbond
x_fit_gocc = np.linspace(X_gocc.min(), X_gocc.max(), 100).reshape(-1, 1)

# Fit a linear regression model for gocc/gbond
linear_model_gocc = LinearRegression()
linear_model_gocc.fit(X_gocc, y_gocc)
y_pred_linear_gocc = linear_model_gocc.predict(x_fit_gocc)
linear_eq_gocc = f"y = {linear_model_gocc.coef_[0]:.2f}x + {linear_model_gocc.intercept_:.2f}"
r2_linear_gocc = r2_score(y_gocc, linear_model_gocc.predict(X_gocc))

# Fit a polynomial regression model for gocc/gbond (degree 2)
poly_gocc = PolynomialFeatures(degree=3)
X_gocc_poly = poly_gocc.fit_transform(X_gocc)
x_fit_gocc_poly = poly_gocc.fit_transform(x_fit_gocc)

poly_model_gocc = LinearRegression()
poly_model_gocc.fit(X_gocc_poly, y_gocc)
y_pred_poly_gocc = poly_model_gocc.predict(x_fit_gocc_poly)
poly_eq_gocc = f"y = {poly_model_gocc.coef_[3]:.2f}x³ + {poly_model_gocc.coef_[2]:.2f}x² + {poly_model_gocc.coef_[1]:.2f}x + {poly_model_gocc.intercept_:.2f}"
r2_poly_gocc = r2_score(y_gocc, poly_model_gocc.predict(X_gocc_poly))

# Filter the rows where epsilon_p [%] < 50 and VEC [e-] < 6
df_VEC = df[(df["epsilon_p [%]"] < 50) & (df["VEC [e-]"] < 6)]

# Define the features and target for VEC [e-]
X_VEC = df_VEC["VEC [e-]"].values.reshape(-1, 1)
y_VEC = df_VEC["epsilon_p [%]"].values

# Create a range of x values for fitting between min and max of VEC [e-]
x_fit_VEC = np.linspace(X_VEC.min(), X_VEC.max(), 100).reshape(-1, 1)

# Fit a linear regression model for VEC [e-]
linear_model_VEC = LinearRegression()
linear_model_VEC.fit(X_VEC, y_VEC)
y_pred_linear_VEC = linear_model_VEC.predict(x_fit_VEC)
linear_eq_VEC = f"y = {linear_model_VEC.coef_[0]:.2f}x + {linear_model_VEC.intercept_:.2f}"
r2_linear_VEC = r2_score(y_VEC, linear_model_VEC.predict(X_VEC))

# Fit a polynomial regression model for VEC [e-] (degree 2)
poly_VEC = PolynomialFeatures(degree=2)
X_VEC_poly = poly_VEC.fit_transform(X_VEC)
x_fit_VEC_poly = poly_VEC.fit_transform(x_fit_VEC)

poly_model_VEC = LinearRegression()
poly_model_VEC.fit(X_VEC_poly, y_VEC)
y_pred_poly_VEC = poly_model_VEC.predict(x_fit_VEC_poly)
poly_eq_VEC = f"y = {poly_model_VEC.coef_[2]:.2f}x² + {poly_model_VEC.coef_[1]:.2f}x + {poly_model_VEC.intercept_:.2f}"
r2_poly_VEC = r2_score(y_VEC, poly_model_VEC.predict(X_VEC_poly))

# Plot the results in two subplots
plt.figure(figsize=(14, 8))

# Subplot 1: gocc/gbond vs epsilon_p [%]
plt.subplot(1, 2, 1)
plt.scatter(df_gocc["gocc/gbond"], df_gocc["epsilon_p [%]"], color="blue", label="Data points")
plt.plot(x_fit_gocc, y_pred_linear_gocc, color="red", label=f"Linear fit: {linear_eq_gocc}, R²={r2_linear_gocc:.2f}")
plt.plot(x_fit_gocc, y_pred_poly_gocc, color="green", label=f"Polynomial fit: {poly_eq_gocc}, R²={r2_poly_gocc:.2f}")
plt.xlabel("gocc/gbond")
plt.ylabel("epsilon_p [%]")
plt.xlim([0.8, 1])
plt.title("gocc/gbond vs epsilon_p [%]")
plt.legend()
plt.grid(True)

# Subplot 2: VEC [e-] vs epsilon_p [%]
plt.subplot(1, 2, 2)
plt.scatter(df_VEC["VEC [e-]"], df_VEC["epsilon_p [%]"], color="blue", label="Data points")
plt.plot(x_fit_VEC, y_pred_linear_VEC, color="red", label=f"Linear fit: {linear_eq_VEC}, R²={r2_linear_VEC:.2f}")
plt.plot(x_fit_VEC, y_pred_poly_VEC, color="green", label=f"Polynomial fit: {poly_eq_VEC}, R²={r2_poly_VEC:.2f}")
plt.xlabel("VEC [e-]")
plt.ylabel("epsilon_p [%]")
plt.xlim([4, 6])
plt.title("VEC [e-] vs epsilon_p [%]")
plt.legend()
plt.grid(True)

# Save and show the combined plot
plt.tight_layout()
plt.savefig("bsd2epsilonFit.png")
plt.show()