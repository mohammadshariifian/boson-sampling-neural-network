# data_loader.py
import os
import pickle
from fractions import Fraction

import numpy as np
import tensorflow as tf
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.join(os.path.expanduser("~"), "BSNN_data")
os.makedirs(DATA_DIR, exist_ok=True)

def prepare_data(data_type):
    X_train, X_test, y_train, y_test = None, None,None,None
    if data_type == 'ionosphere':
        X_data, y_data = ionosphere_data()
    elif data_type == 'spambase':
        X_data, y_data = spambase_data()
    elif data_type == 'MNIST':
        X_train, X_test, y_train, y_test = mnist_prepare_data()
        X_data = None
        y_data = None
    elif data_type == 'fashionMNIST':
        X_train, X_test, y_train, y_test = fashion_mnist_prepare_data()
        X_data = None
        y_data = None
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

        
    return X_data, y_data, X_train, X_test, y_train, y_test




def spambase_data():
    # Load Spambase dataset from OpenML
    data = fetch_openml(name="spambase", version=1, as_frame=False)
    X = data.data.astype(np.float32)
    y = data.target.astype(int)

    # Min-max normalization per feature
    for i in range(X.shape[1]):
        feature_min = X[:, i].min()
        feature_max = X[:, i].max()
        if feature_max > feature_min:
            X[:, i] = (X[:, i] - feature_min) / (feature_max - feature_min)
        else:
            X[:, i] = 0.0  # Handle constant feature

    X_tensor = torch.from_numpy(X).float()
    X_data = [x.clone() for x in X_tensor]  # List of 1D tensors (rows)

    y_data = torch.tensor(y, dtype=torch.int64)
    return X_data, y_data


def mnist_prepare_data():
    dataset_path = os.path.join(DATA_DIR, "mnist_data.pkl")

    # Check if the dataset is already downloaded and saved
    if not os.path.exists(dataset_path):
        print("Downloading and saving MNIST dataset...")
        # Download the MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Save the dataset to a file
        with open(dataset_path, 'wb') as f:
            pickle.dump(((x_train, y_train), (x_test, y_test)), f)
        print("Dataset saved successfully.")
    else:
        # Load the dataset from the saved file
        with open(dataset_path, 'rb') as f:
            (x_train, y_train), (x_test, y_test) = pickle.load(f)

    # Normalize the pixel values to the range [0, 1]
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    return x_train, x_test, y_train, y_test


def fashion_mnist_prepare_data():
    dataset_path = os.path.join(DATA_DIR, "fashion_mnist_data.pkl")

    # Check if the dataset is already downloaded and saved
    if not os.path.exists(dataset_path):
        print("Downloading and saving Fashion MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        with open(dataset_path, 'wb') as f:
            pickle.dump(((x_train, y_train), (x_test, y_test)), f)
        print("Dataset saved successfully.")
    else:
        with open(dataset_path, 'rb') as f:
            (x_train, y_train), (x_test, y_test) = pickle.load(f)

    # Flatten and normalize
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    return x_train, x_test, y_train, y_test

def ionosphere_data():
    """Load ionosphere data and download it if not present locally."""
    # Define data file path
    FILE_PATH = os.path.join(DATA_DIR, "ionosphere_data.pkl")

    # Check if the dataset exists locally
    if not os.path.exists(FILE_PATH):
        # If dataset doesn't exist, fetch it
        print("Dataset not found. Downloading dataset...")

        # Fetch the ionosphere dataset
        ionosphere = fetch_ucirepo(id=52)

        # Extract features and targets
        X = ionosphere.data.features
        y = ionosphere.data.targets

        # Convert y from DataFrame to Series
        y = y.iloc[:, 0]  # Convert y to a Series

        # Save the dataset to a file
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(FILE_PATH, "wb") as f:
            pickle.dump((X, y), f)
        print(f"Dataset saved to {FILE_PATH}")
    else:
        # If dataset exists, load it
        print(f"Dataset found at {FILE_PATH}. Loading...")
        with open(FILE_PATH, "rb") as f:
            X, y = pickle.load(f)

    # Convert X and y to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.map({'g': 1, 'b': -1}).values, dtype=torch.int64)

    # Step 1: Calculate min and max for each feature (column-wise)
    mins = [torch.min(X_tensor[:, feature]) for feature in range(X_tensor.shape[1])]
    maxs = [torch.max(X_tensor[:, feature]) for feature in range(X_tensor.shape[1])]

    # Step 2: Normalize the data to [0, 1] range for each feature
    X_normalized = torch.zeros_like(X_tensor)
    for feature in range(X_tensor.shape[1]):
        X_normalized[:, feature] = (X_tensor[:, feature] - mins[feature]) / (maxs[feature] - mins[feature] + 1e-8)  # Avoid division by zero

    # Optional: convert each sample (row) to a separate tensor in a list
    X_data = [X_normalized[i] for i in range(X_normalized.shape[0])]

    return X_data, y_tensor
    

def get_or_make_mnist_sep_indices(folder_path, data_type, N, test_portion, jj,
                                 n_train_total, n_test_total, seed_base):
    """
    Returns:
      idx_train_in_trainset: indices into X_train/y_train
      idx_test_in_testset:   indices into X_test/y_test
    """
    path = split_cache_path_mnist(folder_path, data_type, N, test_portion, jj)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["idx_train_in_trainset"], obj["idx_test_in_testset"]

    n_test = int(N * test_portion)
    n_train = N - n_test
    if n_train > n_train_total:
        raise ValueError(f"Need n_train={n_train} but train has {n_train_total}")
    if n_test > n_test_total:
        raise ValueError(f"Need n_test={n_test} but test has {n_test_total}")

    rng = np.random.RandomState(seed_base + int(jj))

    idx_train = rng.choice(np.arange(n_train_total), size=n_train, replace=False).astype(np.int64)
    idx_test  = rng.choice(np.arange(n_test_total),  size=n_test,  replace=False).astype(np.int64)

    with open(path, "wb") as f:
        pickle.dump({
            "idx_train_in_trainset": idx_train,
            "idx_test_in_testset": idx_test,
            "seed_base": seed_base,
            "jj": int(jj),
            "N": int(N),
            "test_portion": float(test_portion),
            "data_type": data_type,
        }, f)

    return idx_train, idx_test

def build_from_mnist_separate(X_train, y_train, X_test, y_test, idx_train, idx_test):
    X_raw_train_np = X_train[idx_train]
    y_train_np     = y_train[idx_train]
    X_raw_test_np  = X_test[idx_test]
    y_test_np      = y_test[idx_test]

    X_raw_np = np.concatenate([X_raw_train_np, X_raw_test_np], axis=0)
    y_labels_np = np.concatenate([y_train_np, y_test_np], axis=0)

    X_raw = [torch.tensor(row, dtype=torch.float32) for row in X_raw_np]
    X_raw_train = X_raw[:len(idx_train)]
    X_raw_test  = X_raw[len(idx_train):]

    y_labels = torch.tensor(y_labels_np, dtype=torch.int64)
    y_train_t = torch.tensor(y_train_np, dtype=torch.int64)
    y_test_t  = torch.tensor(y_test_np, dtype=torch.int64)

    subset_indexes_train = np.arange(0, len(idx_train), dtype=np.int64)
    subset_indexes_test  = np.arange(len(idx_train), len(idx_train) + len(idx_test), dtype=np.int64)

    # IMPORTANT: keep “global” indices meaningful:
    # refer to original split + index (not one concatenated pool)
    global_indexes_train = idx_train.copy()
    global_indexes_test  = idx_test.copy()

    return (X_raw, X_raw_train, X_raw_test, y_labels, y_train_t, y_test_t,
            subset_indexes_train, subset_indexes_test, global_indexes_train, global_indexes_test)


def build_from_global_split(X_pool, y_pool, idx_subset, idx_train_global, idx_test_global):
    """
    Build X_raw (list of tensors) and train/test splits from GLOBAL indices into a single pool.
    Returns:
      X_raw, X_raw_train, X_raw_test, y_labels, y_train, y_test,
      subset_indexes_train, subset_indexes_test, global_indexes_train, global_indexes_test
    """
    # X_raw in a fixed order = idx_subset order
    if isinstance(X_pool, np.ndarray):
        X_raw = [torch.tensor(X_pool[i], dtype=torch.float32) for i in idx_subset]
    else:
        # list/torch tensor case
        X_raw = [X_pool[i].detach().cpu() if torch.is_tensor(X_pool[i]) else torch.tensor(X_pool[i], dtype=torch.float32)
                for i in idx_subset]

    # labels aligned with idx_subset order
    if isinstance(y_pool, np.ndarray):
        y_labels = torch.tensor(y_pool[idx_subset], dtype=torch.int64)
    elif torch.is_tensor(y_pool):
        y_labels = y_pool[idx_subset].to(dtype=torch.int64).detach().cpu()
    else:
        y_labels = torch.tensor([y_pool[i] for i in idx_subset], dtype=torch.int64)

    # global -> local mapping (fast via sorting)
    idx_subset = np.asarray(idx_subset, dtype=np.int64)
    order = np.argsort(idx_subset)
    sorted_subset = idx_subset[order]

    idx_train_global = np.asarray(idx_train_global, dtype=np.int64)
    idx_test_global  = np.asarray(idx_test_global, dtype=np.int64)

    idx_train_local = order[np.searchsorted(sorted_subset, idx_train_global)]
    idx_test_local  = order[np.searchsorted(sorted_subset, idx_test_global)]

    X_raw_train = [X_raw[i] for i in idx_train_local]
    X_raw_test  = [X_raw[i] for i in idx_test_local]
    y_train = y_labels[idx_train_local]
    y_test  = y_labels[idx_test_local]

    subset_indexes_train = idx_train_local
    subset_indexes_test  = idx_test_local
    global_indexes_train = idx_train_global
    global_indexes_test  = idx_test_global

    return (X_raw, X_raw_train, X_raw_test, y_labels, y_train, y_test,
            subset_indexes_train, subset_indexes_test, global_indexes_train, global_indexes_test)

def split_cache_path(folder_path, data_type, N, test_portion, jj):
    # Cache depends ONLY on (dataset, N, test_portion, jj)
    portion = Fraction(test_portion).limit_denominator()
    return os.path.join(
        folder_path,
        "splits",
        f"split_{data_type}_N{N}_test{portion.numerator}of{portion.denominator}_jj{jj}.pkl"
    )

def split_cache_path_mnist(folder_path, data_type, N, test_portion, jj):
    portion = Fraction(test_portion).limit_denominator()
    return os.path.join(
        folder_path, "splits",
        f"split_{data_type}_N{N}_test{portion.numerator}of{portion.denominator}_jj{jj}_mnistsep.pkl"
    )

def get_or_make_split_indices(folder_path, data_type, N, test_portion, jj, total_len, seed_base):
    """
    Returns (idx_subset_global, idx_train_global, idx_test_global)
    where all are numpy int64 arrays indexing into X_data / y_data.
    """
    path = split_cache_path(folder_path, data_type, N, test_portion, jj)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["idx_subset_global"], obj["idx_train_global"], obj["idx_test_global"]

    # Deterministic per jj (and N/test_portion/data_type via filename)
    rng = np.random.RandomState(seed_base + int(jj))

    if N > total_len:
        raise ValueError(f"N={N} > total_len={total_len} for data_type={data_type}")

    idx_all = np.arange(total_len, dtype=np.int64)

    if N == total_len:
        idx_subset = idx_all
    else:
        idx_subset = rng.choice(idx_all, size=N, replace=False)

    idx_train, idx_test = train_test_split(
        idx_subset,
        test_size=test_portion,
        random_state=seed_base + int(jj),
        stratify=None  
    )

    idx_subset = np.array(idx_subset, dtype=np.int64)
    idx_train  = np.array(idx_train,  dtype=np.int64)
    idx_test   = np.array(idx_test,   dtype=np.int64)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "idx_subset_global": idx_subset,
                "idx_train_global": idx_train,
                "idx_test_global": idx_test,
                "seed_base": seed_base,
                "jj": int(jj),
                "N": int(N),
                "test_portion": float(test_portion),
                "data_type": data_type,
            },
            f
        )
    return idx_subset, idx_train, idx_test

