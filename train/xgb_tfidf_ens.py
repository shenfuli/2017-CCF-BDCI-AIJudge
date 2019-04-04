# -*- encoding:utf-8 -*-
import sys

sys.path.append("D:/ML_Study/2017-CCF-BDCI-AIJudge")

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from utils import LOGGER
from config.db_config import Config

import warnings
warnings.filterwarnings('ignore')
config = Config()

def main():
    print("hello")

if __name__ == "__main__":
    main()