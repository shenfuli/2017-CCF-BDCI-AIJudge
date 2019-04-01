# -*- encoding:utf-8 -*-
import sys
sys.path.append("/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge")

import codecs
from collections import namedtuple

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from utils import LOGGER
from config.db_config import Config


