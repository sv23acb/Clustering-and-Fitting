# Clustering-and-Fitting
The python code provides clustering and machine learning process on Lung cancer dataset.  
The dataset includes data on several variables that may be linked to lung cancer, such as age, gender, smoking habits, yellow finger presence, anxiety levels, peer pressure, chronic illnesses, exhaustion, allergies, wheezing, alcohol consumption, coughing, shortness of breath, difficulty swallowing, chest pain, and lung cancer presence (target variable). There are 24 examples in the dataset, with both male and female participants. 
#Required python libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
