import numpy as np

from common.utils.common_utils import get_class_from_mask


def intersection_over_union(
    prediction: np.ndarray, ground_truth: np.ndarray, class_label: int = 1
):
    SMOOTH = 1e-6

    prediction = prediction.squeeze()

    assert prediction.shape == ground_truth.shape

    prediction, ground_truth = get_class_from_mask(
        prediction, ground_truth, class_label=class_label
    )

    intersection = (prediction & ground_truth).astype("float32").sum()
    union = (prediction | ground_truth).astype("float32").sum()

    return (intersection + SMOOTH) / (union + SMOOTH)


def accuracy(prediction: np.ndarray, ground_truth: np.ndarray, class_label: int = 1):
    prediction = prediction.squeeze()

    assert prediction.shape == ground_truth.shape

    prediction, ground_truth = get_class_from_mask(
        prediction, ground_truth, class_label=class_label
    )

    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()

    return (prediction == ground_truth).astype("float32").sum() / len(prediction)


def presicion(prediction: np.ndarray, ground_truth: np.ndarray, class_label: int = 1):
    SMOOTH = 1e-6

    prediction = prediction.squeeze()

    assert prediction.shape == ground_truth.shape

    prediction, ground_truth = get_class_from_mask(
        prediction, ground_truth, class_label=class_label
    )

    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()

    TP = ((prediction == 1) & (ground_truth == 1)).sum()
    FP = ((prediction == 1) & (ground_truth == 0)).sum()
    return (TP + SMOOTH) / (TP + FP + SMOOTH)


AVAILABLE_METRICS = {
    "intersection_over_union": intersection_over_union,
    "accuracy": accuracy,
    "presicion": presicion,
}
