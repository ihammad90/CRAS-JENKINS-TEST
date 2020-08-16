# import pandas as pd
# import numpy as np
# #import seaborn as sns
# from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import r2_score
# from sklearn.ensemble import RandomForestRegressor
# import pickle
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from fbprophet import Prophet
# from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
# import config
# import os

# os.system('cmd /c "cls"')







from pipeline import Pipeline

pipeline = Pipeline()

if __name__ == '__main__':

	pipeline.scoring_orchestrator()

	# # load process table
	# log = pd.read_csv(config.Model_backlog)

	# # loading dataset
	# data = pd.read_csv(config.PATH_TO_DATASET)

	# # loading parameters table
	# hyp_set = pd.read_csv('ParameterTable.csv')

	# ############################################

	# # calling function to create scoring data profile
	# pipeline.score_data_profile(log)

	# # calling function to load active models make predictions and save their performance in DB
	# pipeline.load_activemodels(log)

	# # calling function to rank models and select champion from them and save final resultant
	# pipeline.rank_performance(log)

	print("Success")
