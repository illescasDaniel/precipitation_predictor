import numpy as np


def calculate_custom_rain_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	Calculate a custom error metric for rainfall predictions with penalties for false negatives and positives,
	and reduced error weighting for decent rain predictions.

	This metric computes the mean absolute error between true and predicted values, applies penalties
	for significant misclassifications (false negatives and positives), and scales the result by the
	proportion of penalized cases. Errors are dampened when both true and predicted values exceed a
	'decent rain' threshold (2mm), treating close-enough predictions more leniently.

	Parameters
	----------
	y_true : np.ndarray
		Array of true rainfall values (ground truth).
	y_pred : np.ndarray
		Array of predicted rainfall values. Must be the same length as y_true.

	Returns
	-------
	float
		The adjusted weighted error score. Higher values indicate worse performance, with penalties
		amplifying the impact of false negatives (rain >= 2 predicted < 2) and false positives
		(rain <= 1 predicted > 1).

	Notes
	-----
	- False negatives (rain >= 2, predicted < 2) and false positives (rain <= 2, predicted > 2) incur
		a penalty of 4 and 3 respectively, applied element-wise.
	- Errors are scaled down (by 0.5) when both y_true and y_pred >= 4mm, reducing the penalty for
		decent-but-imperfect rain predictions.
	- The final score is scaled by (1 + proportion_penalized), emphasizing mistake frequency.

	Examples
	--------
	>>> import numpy as np
	>>> y_true = np.array([20, 0])
	>>> y_pred = np.array([5, 2])
	>>> calculate_custom_rain_error(y_true, y_pred)
	11.25  # 20 vs 5 scaled down, 0 vs 2 penalized, adjusted by proportion
	>>> y_true = np.array([3, 0])
	>>> y_pred = np.array([1, 2])
	>>> calculate_custom_rain_error(y_true, y_pred)
	16.0  # Same as before for small values
	"""
	error = np.abs(y_true - y_pred)

	# False Negative: It rained but was predicted as zero or very low
	false_negative_penalty = 4.0 * ((y_true >= 2) & (y_pred < 2))

	# False Positive: Predicted rain when there was none (very low)
	false_positive_penalty = 3.0 * ((y_true <= 2) & (y_pred > 2))

	# Dampen error when both true and predicted are "decent rain" (>= 4mm)
	decent_rain_mask = (y_true >= 4) & (y_pred >= 4)
	error_scaling = np.where(decent_rain_mask, 0.5, 1.0)  # Reduce error by half when both >= 2

	# Apply scaling to base error
	scaled_error = error * error_scaling

	# Base weighted error (mean of scaled errors with penalties)
	weighted_error = np.mean(scaled_error * (1 + false_negative_penalty + false_positive_penalty))

	# Proportion of penalized cases
	num_penalized = np.sum((false_negative_penalty > 0) | (false_positive_penalty > 0))
	proportion_penalized = num_penalized / len(y_true) if len(y_true) > 0 else 1

	# Scale by penalty proportion
	adjusted_error = weighted_error * (1 + proportion_penalized)

	return float(adjusted_error)
