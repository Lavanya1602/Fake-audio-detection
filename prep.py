from sklearn.model_selection import train_test_split

data_dir = '/content/sample_data/dataset'
features, labels = load_dataset(data_dir)
features = np.expand_dims(features, axis=-1)  # Add a channel dimension
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
