import os

# Data
DATASET_PATH = os.path.join("datasets", "dataset_cell_growth_4factor.csv")

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 0

# Cross-validation
CV_FOLDS = 5

# Hyperparameter grids
POLY_DEGREES = [1, 2, 3, 4, 5, 6]
KNN_NEIGHBORS = list(range(1, 41))