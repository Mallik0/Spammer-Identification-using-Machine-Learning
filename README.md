
# Spam Detection Using Machine Learning

## Overview

This project implements a **Spam Detection** system using the **Naive Bayes** classification algorithm. It is integrated with a **Tkinter** GUI to allow users to classify emails or text messages as either spam or not spam. The project leverages **regular expressions**, file input/output operations, and machine learning model handling.

## Features

- **Spam Detection**: Classifies text data as spam or not using a trained Naive Bayes classifier.
- **GUI Interface**: User-friendly interface built with Tkinter.
- **Regular Expressions**: Preprocessing step to clean and format data before classification.
- **Model Persistence**: The trained model is saved and loaded using serialization (e.g., pickle).
  
## Requirements

To run the project, ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include the following:

```txt
numpy
pandas
scikit-learn
tkinter
```

## Usage

1. Train the model by providing a dataset of spam and non-spam messages. The model will be serialized and saved for future use.
2. Start the application:
3. Use the GUI to input a message and get the spam classification result.

## Project Structure

```plaintext
├── app.py              # Main application script with Tkinter GUI
├── model.py            # Naive Bayes classifier and text preprocessing
├── data/               # Directory for storing datasets (CSV files)
├── saved_model.pkl     # Serialized Naive Bayes model
├── README.md           # Project documentation
└── requirements.txt    # Python package dependencies
```



