def detect_fake(audio_file_path, model):
    """Classify if the audio is fake or real."""
    features = extract_features(audio_file_path)
    if features is not None:
        features = np.expand_dims(features, axis=[0, -1])  # Prepare for prediction
        prediction = model.predict(features)
        return "Fake" if prediction >= 0.5 else "Real"
    return "Error processing file"

audio_path = '/content/sample_data/dataset/real/real_10000.wav'  # enter the path of your sample data.
print(f"The audio is {detect_fake(audio_path, model)}")
