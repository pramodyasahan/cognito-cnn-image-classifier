# Step 1: Install required libraries (only once per Colab runtime)
!pip install tensorflow scikit-learn matplotlib pandas

# Step 2: Import dependencies
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Import dataset and tools from scikit-learn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
