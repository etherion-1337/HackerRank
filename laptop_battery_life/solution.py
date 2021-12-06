#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def lr_model(train_path):
    train_data = pd.read_csv(train_path, names=["charged", "lasted"])
    train_data_sliced = train_data[train_data["lasted"]<8]
    model = LinearRegression()
    model.fit(train_data_sliced['charged'].values.reshape(-1, 1), train_data_sliced['lasted'].values.reshape(-1, 1))
    return model
    
if __name__ == '__main__':
    timeCharged = float(input())
    model = lr_model("trainingdata.txt")
    ans = model.predict([[timeCharged]])
    print(min(ans[0][0], 8))