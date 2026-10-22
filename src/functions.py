import pandas as pd
from src.constants import CV_FOLDS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
import numpy as np




def load_data(filepath):
    """load the dataset and split it into features (x) and target (y)"""

    df = pd.read_csv(filepath)

    target = df.columns[0]
    features = df.columns[1:]

    x = df.loc[:, features].to_numpy()
    y = df.loc[:, target].to_numpy()

    return x, y, features, target
def tune_polynomial(x_train, y_train, degrees):
    """Find the best polynomial degree using 5-fold cross validation"""

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression(),
    )

    param_grid = dict(polynomialfeatures__degree=degrees)

    model_gscv = GridSearchCV(model, param_grid=param_grid, cv=5,
                              scoring="neg_root_mean_squared_error",
                              return_train_score=True)
    model_gscv.fit(x_train, y_train)

    return model_gscv

def tune_knn(x_train, y_train, n_neighbors):
    """Find the best knn degree using 5-fold cross validation"""
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(),
    )

    param_grid = dict(
        kneighborsregressor__n_neighbors=n_neighbors,
        kneighborsregressor__weights=["uniform", "distance"],
    )
    model_gscv = GridSearchCV(model, param_grid=param_grid, cv=5,
                              scoring="neg_root_mean_squared_error",
                              return_train_score=True)
    model_gscv.fit(x_train, y_train)

    return model_gscv
def plot_rmse_curves(x_values, curves, xlabel, filepath):
    """Plot one or RMSE curves and save the figure"""
    fig, ax= plt.subplots(figsize=(6, 4.5))

    for label, rmse in curves.items():
        ax.plot(x_values, rmse, marker="o", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (d$^{-1}$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
def plot_response_curves(model, x_train, features, filepath):
    """Plot the predicted growth rate when one feature varies and the others stay at their medium"""
    medians = np.median(x_train, axis=0)

    fig, axs = plt.subplots(2,2, figsize=(10, 8))

    for i, ax in enumerate(axs.flat):
        values = np.linspace(x_train[:, i].min(), x_train[:, i].max(), 100)

        x_grid = np.tile(medians, (100,1))
        x_grid[:, i]= values

        ax.plot(values, model.predict(x_grid))
        ax.set_xlabel(features[i])
        ax.set_ylabel("Predicted growth rate (d$^{-1}$)")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)