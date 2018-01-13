# Kaggle Speech Recognition Challenge

This is my entry for the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).

Files are numbered in order I created them, not necessarily the order to run them. Final pipeline:

1. `05_run_opensmile.py` to run OpenSmile voice activity detection for all the files
2. `06_max_vad.py` to extract the max from each OpenSmile output
3. `GCommandsPytorch/02_train_all_no_crossvalidate.py` to train neural network on all the training data
4. `GCommandsPytorch/04_predict_with_prob.py` to predict using neural network
5. `03b_postprocess_unknown_threshold.py` to combine VAD and CNN output to create final predictions
