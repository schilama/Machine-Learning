import numpy as np


def load_data(data_path, to_dense=True):
    """Load data from the compressed NPZ file.

    Arguments:
        data_path: path to the data file.
        to_dense: if X is in the coordinate sparse format (COO), convert it
            to a dense ndarray when setting to_dense=True.

    Returns:
        X (numpy ndarray, shape = (samples, 3600)):
            Training input matrix where each row is a feature vector.
        y (numpy ndarray, shape = (samples,)):
            Training labels. Each entry is either 0 or 1.
        offset (numpy ndarray, shape = (samples, 2)):
            The (y, x) coordinate of the top-left corner of the
            28x28 bounding box where the MNIST digit is located.
    """
    data = np.load(data_path)
    X = data['X']
    if X.size == 1:
        X = data['X'][()]
        if to_dense:
            X = X.toarray()
    y = data['y']
    offset = data['offset']
    return X, y, offset


def to_dense(X):
    """Convert a sparse (COO) matrix to a dense numpy ndarray."""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X


def reshape_to_4d(X, image_size=(60, 60)):
    """Reshape the input matrix into 4D representation:
    (N, channels, image_height, image_width)

    This is useful for applying convolutional layers and so on.
    """
    return X.reshape(X.shape[0], 1, image_size[0], image_size[0])
