import os

def load_dataset(data_dir):
    """Load dataset from the specified directory."""
    features, labels = [], []
    for label, sub_dir in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
        sub_dir_path = os.path.join(data_dir, sub_dir)
        for file_name in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file_name)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)
    return np.array(features), np.array(labels)
