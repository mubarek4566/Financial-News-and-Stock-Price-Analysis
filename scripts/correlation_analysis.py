import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CorrelationAnalysis:
    def __init__(self, path):
        self.df = path

    def sentiment_analysis(self):
        