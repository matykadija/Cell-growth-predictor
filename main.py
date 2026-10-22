import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.functions import load_data, tune_polynomial, tune_knn, plot_rmse_curves, plot_response_curves
from src.constants import DATASET_PATH, TEST_SIZE, RANDOM_STATE, POLY_DEGREES, KNN_NEIGHBORS

#to load data
x, y, features, target = load_data(DATASET_PATH)

#to split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"Features: {list(features)}")
print(f"Target: {target}")
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

#Polynomial regression
poly_gscv = tune_polynomial(x_train, y_train, degrees=POLY_DEGREES)

poly_results = pd.DataFrame(poly_gscv.cv_results_)
print("\nPolynomial regression:")
for _, row in poly_results.iterrows():
    print(f"  degree{row['param_polynomialfeatures__degree']}  | "
          f"train RMSE {-row['mean_train_score']:.3f} | "
          f"valid RMSE {-row['mean_test_score']:.3f} | ")

print(f"Best: {poly_gscv.best_params_}")

# k-nearest neighbors regression
knn_gscv = tune_knn(x_train, y_train, n_neighbors=KNN_NEIGHBORS)

print(f"\nkNN regression:")
print(f"Best: {knn_gscv.best_params_}")
print(f"Best valid RMSE: {-knn_gscv.best_score_:.3f}")

#Export result tables
poly_table = pd.DataFrame({
    "degree": poly_gscv.cv_results_["param_polynomialfeatures__degree"],
    "train_rmse": -poly_results["mean_train_score"],
    "valid_rmse": -poly_results["mean_test_score"],
}).round(3)
poly_table.to_csv(os.path.join("tables", "polynomial_results.csv"), index=False)

comparaison = pd.DataFrame({
    "model": [f"Polynomial {poly_gscv.best_params_}", f"KNN {knn_gscv.best_params_}"],
    "valid_rmse": [-poly_gscv.best_score_, -knn_gscv.best_score_],
}).round(3)
comparaison.to_csv(os.path.join("tables", "model_comparison.csv"), index=False)

# Figures

poly_plot = poly_table[poly_table["degree"] <= 5]
plot_rmse_curves(
    poly_plot["degree"],
    {"Training": poly_plot["train_rmse"], "Validation": poly_plot["valid_rmse"]},
    "Polynomial degree",
    os.path.join("figures", "polynomial_validation_curve.png"),
)
knn_results = pd.DataFrame(knn_gscv.cv_results_)
knn_uniform = knn_results[knn_results["param_kneighborsregressor__weights"] == "uniform"]
knn_distance = knn_results[knn_results["param_kneighborsregressor__weights"] == "distance"]
plot_rmse_curves(
    KNN_NEIGHBORS,
    {"Validation (uniform)": -knn_uniform["mean_test_score"].to_numpy(),
            "Validation (distance)": -knn_distance["mean_test_score"].to_numpy()},
    "Number of neighbors (k)",
    os.path.join("figures", "knn_validation_curve.png"),
)

#Final evaluation of the selected model on the test set
final_model = poly_gscv.best_estimator_
test_rmse = -poly_gscv.score(x_test, y_test)

print(f"\nSelected model: polynomial degree {poly_gscv.best_params_['polynomialfeatures__degree']}")
print(f"Validation RMSE: {-poly_gscv.best_score_:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

#Effect of each variable on the predicted growth rate
plot_response_curves(poly_gscv.best_estimator_, x_train, features,
                     os.path.join("figures", "response_curves.png"))


