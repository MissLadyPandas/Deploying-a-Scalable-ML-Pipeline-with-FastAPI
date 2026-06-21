import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data


@pytest.fixture
def sample_data():
    """ Small synthetic dataset mimicking the census data structure, for fast tests. """
    data = pd.DataFrame({
        "age": [25, 38, 45, 33, 52, 29],
        "workclass": ["Private", "Self-emp", "Private", "Govt", "Private", "Govt"],
        "education": ["Bachelors", "HS-grad", "Masters", "Bachelors", "HS-grad", "Masters"],
        "salary": ["<=50K", "<=50K", ">50K", "<=50K", ">50K", ">50K"],
    })
    return data


def test_train_model_returns_random_forest(sample_data):
    """
    Test that train_model returns a fitted RandomForestClassifier instance.
    """
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output_type_and_shape(sample_data):
    """
    Test that inference returns a numpy array with one prediction per input row.
    """
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_known_values():
    """
    Test that compute_model_metrics returns the correct precision, recall, and F1
    for a known set of labels and predictions.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == pytest.approx(0.6667, abs=0.001)
