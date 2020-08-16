import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from datetime import datetime
from datetime import date
from time import time
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
import glob

import config
import os

os.system('cmd /c "cls"')

root = Path('.')


class Pipeline:
	def __init__(self):


		self.df_list = None
		self.train_df_list = None
		self.test_df_list = None
		self.Ex_time = None



	#below function does imputation by filling nulls with mean of the column
	def Imputation(self, df,var_table):
		#loop through dataframe and check for nulls in each column
		Threshold = 0
		variables = var_table[(var_table.Variable_Class != 'SKU') & (var_table.Variable_Class != 'Region')]['Variable'].tolist()
		dff = df[[variables]]
		for (columnName, columnData) in dff.iteritems():
			Threshold = var_table.loc[var_table.Variable == columnName]['Null_Threshold'].tolist()[0]
			temp = dff[[columnName]]
			#if nulls is less than the threshold then fill all the nulls of column with mean of the column
			if temp.isnull().sum() < Threshold:
				temp = fillna(temp.mean())
				df[columnName] = temp
		#return the dataframe after imputation
		return df
	
	#remove nulls	
	def Remove_Null(self,df):
		#remove all the nulls and return the dataframe

		df = df.dropna()
		return df

	# def Data_Aggregation(self, df):
		#Aggregate data on specific level according to grain
		#Values of Media variables will remain same
		#Price will be same
		#Incentives will add up
		#above Approach is valid if aggregation is being done on category level or product level for same period
		#if aggregation is done on according to grain of period then approach will be changed

	# def var_encoding(self,df):
	# 	dum_df = pd.get_dummies(df, columns=[var_encode] )
	# 	# merge with main df bridge_df on key values
	# 	df = df.join(dum_df)
	# 	return df

	# def varaible_scaling(self,df):
	# 	ss = StandardScaler()
	# 	d = df[reduced_Var]
	# 	df_scaled = pd.DataFrame(ss.fit_transform(df),columns = reduced_Var)
	# 	return df_scaled

	#def Drived_Variables(self,df):
		#we will create variables if needed in this function

	#def feature_selection(self,df):
		#Feature Selection will be done in this function and selected
		#features will be written in variable table.

	#This function checks for erroneous_column and returns column name containing error
	def Erroneous_Column(self,data,var_table):

		variables = var_table[(var_table.Variable_Class != 'SKU') & (var_table.Variable_Class != 'Region')]['Variable'].tolist()
		variables = list(set(variables))
		
		df = data.copy()

		#make a list of erroneous columns
		Err_col = []
		error_code = 202
		Threshold = 0.5
		#Iterate through the dataframe and calculate percentage null points in each columns
		for (columnName, columnData) in df.iteritems():
			if columnName in variables:
				Threshold = var_table.loc[var_table.Variable == columnName]['Null_Threshold'].tolist()[0]
			else:
				Threshold = 0.5
			temp = df[[columnName]]
			Per_Nulls = temp.isnull().sum()[0]/temp.shape[0]
			#if the percentage of null points is greater than the threshold mark it as erroneous column
			if Per_Nulls > Threshold:
				Err_col.append(columnName)
		
		#return the erroneous columns if exist
		return Err_col

	#This function removes the erroneous_column and returns dataframe after removal of the erroneous column
	def Remove_Err_column(self,df, col):
		#drop the erroneous column and return dataframe
		df = df.drop(col,axis = 1)
		return df

	#This function checks for empty dataset and returns true if dataset is empty
	def Empty_Dataset(self, df):
		#reeturn true if the dataset is empty
		return len(df.index) == 0

#'PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'
	def Error_log(self,log):
		#name of the table
		file_name  =  "Error_log.csv"
		check = 0
		#check if error_log table exists
		if os.path.exists(file_name):
			pass
		#if error_log table is missing then make error_log table
		else:
			temp = pd.DataFrame(columns = ['PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'])
			temp.to_csv('Error_log.csv', index = False)
		#load error_table
		temp = pd.read_csv('Error_log.csv')
		PackageName = self.PackageName
		UserStory = self.UserStory
		ErrorType = self.ErrorType
		ExecutionTime = str(datetime.now().time())
		processID = log.iloc[log.shape[0]-1]['ProcessID']
		#check status if any error occured
		if self.status != None:
			temp = temp.append({'PackageName':PackageName,'UserStory':UserStory,'ErrorType':ErrorType,'ExecutionTime':ExecutionTime,'ProcessID':processID},ignore_index=True)
			check = 1 #error status 1
			#update the error_log
			#print(temp.columns)
		temp.to_csv('Error_log.csv', index = False)
		return self

#'LogID','Date','PackageName','UserStory','Status','StartingTime','ExecutionTime','ProcessID','Region','SKU','algorithm'
	#below functions creates a log_table for mantaining logs of preprocessing 
	def Log_Table(self,log):
		#name of the log_table
		file_name = "Log_Table.csv"
		check = 0
		#check if log_table exists
		if os.path.exists(file_name):
			check = 1
			pass
		else:
			temp = pd.DataFrame(columns =['LogID','Date','PackageName','UserStory','Status','StartingTime','ExecutionTime','ProcessID','Region','SKU','algorithm'])
			#if the log_table does't exist assign logid 1 when creating log_table
			logid = 1
		if check == 1:
			#if path exists read the log_table
			temp = pd.read_csv('Log_Table.csv')
			logid = temp.iloc[temp.shape[0]-1]['LogID']
		p = log.iloc[log.shape[0]-1]['ProcessID']#processID
		SKU = log.iloc[log.shape[0]-1]['SKU']#SKU
		Region = log.iloc[log.shape[0]-1]['Region']#Region
		Algo = log.iloc[log.shape[0]-1]['Models']#Algorithm
		dat = str(date.today())#current date
		st_time = str(datetime.now().time())#current time
		exe_time = self.Ex_time#execution time of preprocessing
		UserStory = self.UserStory
		PackageName = self.PackageName
		status = self.status


		#append the log table with the updated log values
		temp = temp.append({'LogID':logid,'Date':dat,'PackageName':PackageName,'UserStory':UserStory,'Status':status,'StartingTime':st_time,'ExecutionTime':exe_time,'ProcessID':p,'Region':Region,'SKU':SKU,'algorithm':Algo},ignore_index=True)
		#write it back
		temp.to_csv('Log_Table.csv',index = False)
#############################################################################
	#This function prepares the raw data without breaking it into subsets
	def Data_Prep(self, data,var_table):

		if self.Empty_Dataset(data):
			self.status = 400
			self.PackageName = 'Data Preparation'
			self.UserStory = 'Data Not Found'
			self.ErrorType = 'Empty Dataset'
			print("Yes")
			return self.status



		#check for erroneous columns
		column = self.Erroneous_Column(data,var_table)
		check = 0
		if len(column) != 0:
			#remove erroneous columns
			data = self.Remove_Err_column(data,column)
			check = 1
			self.status = 201
			self.PackageName = 'Data Preparation'
			self.UserStory = 'Error in column'
			self.ErrorType = 'Nulls in column'
		#remove nulls once removed erroneous columns
		data = self.Remove_Null(data)
		#return Dataset
		return data
###############################################

	#prepare training and testing datasets
	def raw_train_set(self,log,var):
		#load ads
		df = pd.read_csv('datads.csv')
		#
		SKU_col = var[var.Variable_Class == 'SKU']['Variable'].tolist()[0]
		Region_Col = var[var.Variable_Class == 'Region']['Variable'].tolist()[0]
		#date
		date = var.loc[var.Variable_Class == 'TimeSeries']['Variable'].tolist()[0]
		#training start date
		log['Training_StartDate'] = pd.to_datetime(log['Training_StartDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]

		#training end date
		log['Training_EndDate'] = pd.to_datetime(log['Training_EndDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]
		#training data
		df[date] = pd.to_datetime(df[date],format = "%m/%d/%Y",errors='coerce')
		# train_data = df.loc[(df[date] >= Training_StartDate) & (df[date] <= Training_EndDate)]
		log = log[['ProcessID','SKU','Region','Models','Training_StartDate','Training_EndDate']]
		log = log.drop_duplicates()
		log = log.reset_index()

		for i in range(0,log.shape[0]):
			SKU = log.SKU[i]
			Region = log.Region[i]
			ProcessID = log.ProcessID[i]
			Models = log.Models[i]
			Training_StartDate = log.Training_StartDate[i]
			Training_EndDate = log.Training_EndDate[i]
			data = df.loc[(df[SKU_col] == SKU) & (df[Region_Col] == Region)]
			data = data.loc[(data[date] >= Training_StartDate) & (data[date] <= Training_EndDate)]
			# data = self.Data_Prep(data,var)
			os.makedirs('./Datasets', exist_ok = True)
			nameofdataset = "PID_"+str(ProcessID)+"_"+SKU+"_"+Region+"_"+Models+"_Raw_Train"+".csv"
			nameofdataset = "./Datasets/"+nameofdataset
			data.to_csv(nameofdataset)
		return self


	#prepare training and testing datasets
	def train_set(self,log,var):
		#load ads
		df = pd.read_csv('datads.csv')
		#
		SKU_col = var[var.Variable_Class == 'SKU']['Variable'].tolist()[0]
		Region_Col = var[var.Variable_Class == 'Region']['Variable'].tolist()[0]
		#date
		date = var.loc[var.Variable_Class == 'TimeSeries']['Variable'].tolist()[0]
		#training start date
		log['Training_StartDate'] = pd.to_datetime(log['Training_StartDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]

		#training end date
		log['Training_EndDate'] = pd.to_datetime(log['Training_EndDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]
		#training data
		df[date] = pd.to_datetime(df[date],format = "%m/%d/%Y",errors='coerce')
		# train_data = df.loc[(df[date] >= Training_StartDate) & (df[date] <= Training_EndDate)]
		log = log[['ProcessID','SKU','Region','Models','Training_StartDate','Training_EndDate']]
		log = log.drop_duplicates()
		log = log.reset_index()

		for i in range(0,log.shape[0]):
			SKU = log.SKU[i]
			Region = log.Region[i]
			ProcessID = log.ProcessID[i]
			Models = log.Models[i]
			Training_StartDate = log.Training_StartDate[i]
			Training_EndDate = log.Training_EndDate[i]
			data = df.loc[(df[SKU_col] == SKU) & (df[Region_Col] == Region)]
			data = data.loc[(data[date] >= Training_StartDate) & (data[date] <= Training_EndDate)]
			data = self.Data_Prep(data,var)
			os.makedirs('./Datasets', exist_ok = True)
			nameofdataset = "PID_"+str(ProcessID)+"_"+SKU+"_"+Region+"_"+Models+"_Train"+".csv"
			nameofdataset = "./Datasets/"+nameofdataset
			data.to_csv(nameofdataset)
		return self


	#test_Set
	def test_set(self,log,var):
		#load ads
		df = pd.read_csv('datads.csv')
		#
		SKU_col = var[var.Variable_Class == 'SKU']['Variable'].tolist()[0]
		Region_Col = var[var.Variable_Class == 'Region']['Variable'].tolist()[0]
		#date
		date = var.loc[var.Variable_Class == 'TimeSeries']['Variable'].tolist()[0]
		#training start date
		log['Testing_StartDate'] = pd.to_datetime(log['Testing_StartDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]

		#training end date
		log['Testing_EndDate'] = pd.to_datetime(log['Testing_EndDate'],format = "%m/%d/%Y",errors='coerce').tolist()[0]
		#training data
		df[date] = pd.to_datetime(df[date],format = "%m/%d/%Y",errors='coerce')
		# train_data = df.loc[(df[date] >= Training_StartDate) & (df[date] <= Training_EndDate)]
		log = log[['ProcessID','SKU','Region','Models','Testing_StartDate','Testing_EndDate']]
		log = log.drop_duplicates()
		log = log.reset_index()

		for i in range(0,log.shape[0]):
			SKU = log.SKU[i]
			Region = log.Region[i]
			ProcessID = log.ProcessID[i]
			Models = log.Models[i]
			Testing_StartDate = log.Testing_StartDate[i]
			Testing_EndDate = log.Testing_EndDate[i]
			data = df.loc[(df[SKU_col] == SKU) & (df[Region_Col] == Region)]
			data = data.loc[(data[date] >= Testing_StartDate) & (data[date] <= Testing_EndDate)]
			data = self.Data_Prep(data,var)
			os.makedirs('./Datasets', exist_ok = True)
			nameofdataset = "PID_"+str(ProcessID)+"_"+SKU+"_"+Region+"_"+Models+"_Test"+".csv"
			nameofdataset = "./Datasets/"+nameofdataset
			data.to_csv(nameofdataset)
		return self

	#Forecast_Set

	def Forecast_Set(self,log,var):
		#load ads
		df = pd.read_csv('datads.csv')
		#
		SKU_col = var[var.Variable_Class == 'SKU']['Variable'].tolist()[0]
		Region_Col = var[var.Variable_Class == 'Region']['Variable'].tolist()[0]
		#date
		date = var.loc[var.Variable_Class == 'TimeSeries']['Variable'].tolist()[0]
		#training start date
		log['Forecasting_StartDate'] = pd.to_datetime(log['Forecasting_StartDate'],format = "%m/%d/%Y",errors='coerce').tolist()

		#training end date
		log['Forecasting_EndDate'] = pd.to_datetime(log['Forecasting_EndDate'],format = "%m/%d/%Y",errors='coerce').tolist()
		#training data
		df[date] = pd.to_datetime(df[date],format = "%m/%d/%Y",errors='coerce')
		# train_data = df.loc[(df[date] >= Training_StartDate) & (df[date] <= Training_EndDate)]
		log = log[['ProcessID','SKU','Region','Models','Forecasting_StartDate','Forecasting_EndDate']]
		log = log.drop_duplicates()
		log = log.reset_index()
		

		for i in range(0,log.shape[0]):
			SKU = log.SKU[i]
			Region = log.Region[i]
			ProcessID = log.ProcessID[i]
			Models = log.Models[i]
			Forcasting_StartDate = log.Forecasting_StartDate[i]
			Forcasting_EndDate = log.Forecasting_EndDate[i]
			data = df.loc[(df[SKU_col] == SKU) & (df[Region_Col] == Region) & (df[date] >= Forcasting_StartDate) & (df[date] <= Forcasting_EndDate)]
			os.makedirs('./Datasets', exist_ok = True)
			nameofdataset = "PID_"+str(ProcessID)+"_"+SKU+"_"+Region+"_"+Models+"_Forecast"+".csv"
			nameofdataset = "./Datasets/"+nameofdataset
			data.to_csv(nameofdataset)
		return self

		#save train and test sets
	def save_train_test_data(self,log,var):
		self.train_set(log,var)
		self.test_set(log,var)
		self.raw_train_set(log,var)


	def error_update(self, func, err):
		# This function takes strings as inputs and generates those strings as error warning

		print('Error found in function ' + func + ':')
		print(err)

	def train_test_val(self, log):

		# Run loop through the data
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:

				# Check if number of columns in train and test data sets are same
				if self.train_df_list[i][j].shape[1] != self.test_df_list[i][j].shape[1]:

					# #print Error, and pass function name
					self.error_update('train_test_val','number of columns for training and testing set donot match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()


				# Check if Region in train and test data sets are same
				if self.train_df_list[i][j].Region.unique() != self.test_df_list[i][j].Region.unique():

					# #print Error, and pass function name
					self.error_update('train_test_val','Regions in test and training data sets do not match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()

				# Check if Region in train and test data sets are same
				if self.train_df_list[i][j].m_sku_desc.unique() != self.test_df_list[i][j].m_sku_desc.unique():

					# print Error, and pass function name
					self.error_update('train_test_val','SKUs in test and training data sets do not match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()


				j += 1
			i += 1


		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract forecast period in months
		fore_period = last_row.iloc[0]['Forecast_Period']

		# Run loop through the data
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:
				# Check if the number of rows are according to forecast period
				if self.test_df_list[i][j].shape[0] > ((fore_period*4)+2) or self.test_df_list[i][j].shape[0] < ((fore_period*4)-2):

					# Print warnings
					print('WARNING: Number of rows not correct: ')
					print('Number of rows found: ')
					print(self.test_df_list[i][j].shape[0])

				j += 1
			i += 1






	def data_profile(self, df, Data_Stage):

		# Separate Numerical and Categorical variables
		variables_numerical = [*self.X_variables_numerical,*self.targetvariable]
		variables_categorical = self.X_variables_categorical

		# print(variables_numerical)
		# print(variables_categorical)

		# Select the columns of only the variables we need
		df_numerical = df[variables_numerical]
		df_categorical = df[variables_categorical]


		# save data profiles for numerical variables in variables
		prof_median = df_numerical.median(numeric_only=bool)
		prof_unique = df_numerical.nunique(dropna=True)
		prof_mean = df_numerical.mean(numeric_only=bool)
		prof_min = df_numerical.min(numeric_only=bool)
		prof_max = df_numerical.max(numeric_only=bool)
		prof_quant_25 = df_numerical.quantile(numeric_only=bool,q=0.25)
		prof_quant_75 = df_numerical.quantile(numeric_only=bool,q=0.75)
		prof_std = df_numerical.std(numeric_only=bool)
		prof_nan_count = len(df_numerical) - df_numerical.count()

		# create a resultant dataframe for numerical variables
		result_numerical = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)
		
		# reset index
		result_numerical.reset_index(level = 0, inplace =True)
		
		#rename columns
		result_numerical = result_numerical.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

		# set to zero, the profiles which are not needed for numerical variables
		result_numerical.loc[:,'Unique_values'] = 0

		# print('nume')
		# print(result_numerical)

		if self.X_variables_categorical.shape[0] != 0:

			# save data profiles for categorical variables in variables
			prof_median_1 = df_categorical.median(numeric_only=bool)
			prof_unique_1 = df_categorical.nunique(dropna=True)
			prof_mean_1 = df_categorical.mean(numeric_only=bool)
			prof_min_1 = df_categorical.min(numeric_only=bool)
			prof_max_1 = df_categorical.max(numeric_only=bool)
			prof_quant_25_1 = df_categorical.quantile(numeric_only=bool,q=0.25)
			prof_quant_75_1 = df_categorical.quantile(numeric_only=bool,q=0.75)
			prof_std_1 = df_categorical.std(numeric_only=bool)
			prof_nan_count_1 = len(df_categorical) - df_categorical.count()



			# create a resultant dataframe for categorical variables
			result_categorical = pd.concat([prof_median_1,prof_unique_1, prof_mean_1, prof_min_1, prof_max_1, prof_quant_25_1, prof_quant_75_1,prof_std_1,prof_nan_count_1], axis=1)
			

			# reset index
			result_categorical.reset_index(level = 0, inplace =True)
			

			# rename columns
			result_categorical = result_categorical.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)


			# set to zero, the profiles which are not needed for numerical variables
			result_categorical.loc[:,'prof_quant_25'] = 0
			result_categorical.loc[:,'prof_quant_75'] = 0
			result_categorical.loc[:,'prof_std'] = 0

			# print('categ')
			# print(result_categorical)



		if self.X_variables_categorical.shape[0] == 0:
			result = result_numerical
		else:

			# Make a final dataframe to be exported
			result = pd.concat([result_numerical, result_categorical], axis = 0)

		# insert current date into the column
		result['Date'] = str(date.today())

		# insert current time into the column
		result['Time'] = str(datetime.now().time())

		# insert backlog id into column
		result['ProcessID'] = self.processID

		# insert Data Stage
		result['Data_Stage'] = Data_Stage

		# insert region and sku columns
		result['Region'] = self.region
		result['SKU'] = self.SKU


		# name the output file
		nameofprofile = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+Data_Stage + ".csv"
		nameofprofile = "./profiles/" + nameofprofile

		# export to csv
		result.to_csv(nameofprofile, encoding='utf-8',index=False)


###############################################################################################################
	#Preprocessing wraper
	def Preprocessing_orchestrator(self,log,Var_table):

		start_time = time.time()

		# Create a directory 'profiles' if doesnt exist already
		os.makedirs('./profiles', exist_ok = True)

		self.save_train_test_data(log,Var_table)
		self.Forecast_Set(log,Var_table)

		totalrows = log.shape[0]
		i = 0
		while i < totalrows:
			self.read_model_backlog(log,i)
			
			raw_df = self.read_raw_train_data()
			processed_df = self.read_processed_train_data()

			self.data_profile(raw_df, "Raw")
			self.data_profile(processed_df, "Processed")

			i = i + 1

			

		self.Ex_time = time.time() - start_time


##############################################################################################################

					####### Modeling Functionalities #######

################# This is the orchestration function of modeling
################# which call all the modeling functionalities in order

	def modeling_orchestrator(self):
		# process metadata is same as log
		os.makedirs('./Modeling_Output', exist_ok = True)

		# reading models/algos from log and saving in self.algos
		log = pd.read_csv(config.Model_backlog)
		log.reset_index(inplace = True)
		totalrows = log.shape[0]
		i=0

		while i<totalrows:

			self.read_model_backlog(log,i)

			traindata = self.fit_models().train_df

			testdata = self.read_testdata()

			result = self.modeling_performance(testdata,'test')

			self.modeling_performance(traindata,'train')

			self.model_repo(result)

			self.evaluate_model(result)

			i=i+1

		train_results = pd.read_csv('./Modeling_Output/Train_Performance.csv')
		test_results = pd.read_csv('./Modeling_Output/Test_Performance.csv')

		self.rank_performance(train_results,'./Modeling_Output/Train_Performance.csv')
		self.rank_performance(test_results,'./Modeling_Output/Test_Performance.csv')

		



################# This Function Reads the Parameters from metadata
################# which we will be passing to algorithms for training

	def read_model_parameters(self):

		# stores parameters metadata in parameter_table
		parameter_table = pd.read_csv(config.parameter_table)

		#read parameters row for Parameter ID of this process
		parameter_table = parameter_table[(parameter_table['ParameterID'] == self.parameter_id)]
		parameter_table.reset_index(inplace = True)
		
		# Storing Model parameter in separate list for this process
		self.model_params=[x for x in parameter_table.loc[0, parameter_table.columns[3:20]].dropna()]

		return self

################# This Function Reads the algorithms for this process
################# from process metadata

	def read_model_backlog(self,log,i):

		#Read desired values from each row

		self.processID = log['ProcessID'][i]
		self.algorithm = log['Models'][i]
		self.SKU = log['SKU'][i]
		self.region = log['Region'][i]
		self.variable_list_id = log['VariableListID'][i]
		self.parameter_id = log['ParameterID'][i]
		self.performancethres_id = log['ThresID'][i]

		# Calling functions to read variables and parameters for this process and algorithm

		self.read_model_parameters()
		self.read_variable_list()
		self.read_performance_thres()
		return self


################# This Function reads the variables 
################# for this process

	def read_variable_list (self):

		# read variable table for variablelist ID of this process
		variable_table = pd.read_csv(config.variable_table)
		variable_table = variable_table[(variable_table['VariableListID'] == self.variable_list_id)]

		# retrieve target variable name
		self.targetvariable = variable_table[(variable_table['Variable_Class'] == 'Target')]['Variable'].values

		# retrieve names of X variables to use in the model of this process
		self.X_variables = variable_table[(variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values
		self.X_variables_numerical = variable_table[(variable_table['Variable_Type'] == 'Numerical') & (variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values
		self.X_variables_categorical = variable_table[(variable_table['Variable_Type'] == 'Categorical') & (variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values


		return self

################# This Function reads the Thresholds 
################# of performance for models of this process

	def read_performance_thres (self):

		# read Performance Thres table for Thres ID of this process
		threstable = pd.read_csv(config.PerformanceThres_Table)
		threstable = threstable[(threstable['ThresID'] == self.performancethres_id)]

		# retrieve active models thres value
		self.active_thres = threstable['Active'].values
		self.active_thres = self.active_thres[0]

		# retrieve errored models thres value
		self.errored_thres = threstable['Errored'].values
		self.errored_thres = self.errored_thres[0]

		return self


################# This Function defines the Arima Model with parameters
################# read from parameters metadata

	def Arima_models(self, df):

		# df contains training data

		#setting variables required by this algo in a list
		variables = [*self.X_variables,*self.targetvariable]
		
		# training model
		df = df[variables]


		df[self.X_variables[0]] = pd.to_datetime(df[self.X_variables[0]])
		df = df.set_index(self.X_variables[0])
		df=df.fillna(0)
		A=auto_arima(df[self.targetvariable[0]],seasonal=self.model_params[0])
		x=A.order
		model=ARIMA(df[self.targetvariable[0]],order=x)
		results=model.fit()


		#creating name of model
		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel


		#saving model
		with open(nameofmodel,'wb') as t:
			pickle.dump(results, t)

		

################# This Function defines the Lasso Model with parameters
################# read from parameters metadata

	def Lasso_Models(self, df):
		# df contains training data
		X = df[self.X_variables]
		Y = df[self.targetvariable]

		lasso = Lasso(max_iter=int(self.model_params[0]),tol=float(self.model_params[1]))
		model = lasso.fit(X,Y)

		#Saving Models into Local Machine
		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)

################# This Function defines the Gradient Boosting Model with parameters
################# read from parameters metadata
		# df contains training data

	def GBOOST_Models(self, df):
		# df contains training data

		X = df[self.X_variables]
		Y = df[self.targetvariable]

		xgb = GradientBoostingRegressor(n_estimators=int(self.model_params[0]))
		model = xgb.fit(X,Y)

		#Saving Models into Local Machine
		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)
		

################# This Function defines the Random Forest Model with parameters
################# read from parameters metadata

	def RF_Models(self,df):
		# df contains training data

		X = df[self.X_variables]
		Y = df[self.targetvariable]

		rfr = RandomForestRegressor(n_estimators = int(self.model_params[0]))
		model = rfr.fit(X,Y)

		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)
		# model returned after training

################# This Function defines the Prophet Model with parameters
################# read from parameters metadata

	def Prophet_Models(self,df):
		# df contains training data

		variables = [*self.X_variables,*self.targetvariable]

		df = df[variables]
		df[self.X_variables[0]] = pd.to_datetime(df[self.X_variables[0]])
		df = df.set_index(self.X_variables[0])
		df = df.resample('W').mean() #make df daily
		df = df.reset_index()
		df.columns = ['ds', 'y']
		df=df.reset_index()
		df=df.drop('index', axis=1)
		fbp = Prophet(daily_seasonality=bool(self.model_params[0]),yearly_seasonality=int(self.model_params[1]),weekly_seasonality=self.model_params[2])
		model = fbp.fit(df)
		# model returned after training

		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)

################# This Function trains all the models for each algo mentioned in
################# process metadata for each Region and SKU

	def fit_models(self):

		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Train"+".csv"
		nameofdataset = "./Datasets/"+nameofdataset

		# loading training datasets
		# with open("Prepared_Train_DataSets", 'rb') as t:
		# 	self.train_df = pickle.load(t)

		self.train_df = pd.read_csv(nameofdataset)

		os.makedirs('./models', exist_ok = True)
		# if algo is Lasso
		if (self.algorithm=='Lasso'):

			self.Lasso_Models(self.train_df)			

			# if algo is Arima
		elif(self.algorithm=='ARIMA'):

			self.Arima_models(self.train_df)

			# if algo is RandomForest
		elif(self.algorithm=='RandomForest'):

			self.RF_Models(self.train_df)
			

		# if algo is GradientBoosting
		elif(self.algorithm=='GradientBoosting'):

			self.GBOOST_Models(self.train_df)
			

		# if algo is Prophet
		elif(self.algorithm=='Prophet'):

			self.Prophet_Models(self.train_df)

		return self

################# This Function calculates MAPE
#################

	def mape(self, df):

		#df is dataframe having actual and predictions

		dff = df
		a = dff['Actual_Tons']
		f = dff['Predicted_Tons']
		MAPE = np.mean(abs((a-f)/a))
		return MAPE

################# This Function calculated Weighted MAPE
#################

	def wmape(self, df):

		#df is dataframe having actual and predictions

		dff = df#[(df.YEAR_c == 2018) & (df.month_c >9)]
		a = dff['Actual_Tons']
		f = dff['Predicted_Tons']
		WMAPE = np.sum(abs(a-f))/np.sum(a)
		return WMAPE

################# This Function calculates Error
#################

	def error(self, a, f):
		error = a - f
		return error



################# This Function moves the created models in
################# the models Repository
	def read_model(self):
		
		nameofmodel = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm
		nameofmodel = "./models/"+nameofmodel


		with open(nameofmodel, 'rb') as t:
			model = pickle.load(t)

		return model

	def read_processed_train_data(self):

		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Train"+".csv"
		nameofdataset = "./Datasets/"+nameofdataset

		test_df = pd.read_csv(nameofdataset)

		return test_df

	def read_raw_train_data (self):

		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Raw_Train"+".csv"
		nameofdataset = "./Datasets/"+nameofdataset

		test_df = pd.read_csv(nameofdataset)

		return test_df


	def read_testdata (self):
		
		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Test"+".csv"
		nameofdataset = "./Datasets/"+nameofdataset

		test_df = pd.read_csv(nameofdataset)

		return test_df

	def read_forecastdata (self):
		
		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Forecast"+".csv"
		nameofdataset = "./Datasets/"+nameofdataset

		forecast_df = pd.read_csv(nameofdataset)

		return forecast_df



	



	def prediction_func(self, df):

		model = self.read_model()
		
		#test_df_list = read_testdata()

		if (self.algorithm == 'Prophet' or self.algorithm == 'ARIMA'):
			# df = df[0][0]
			df = pd.DataFrame(df[self.X_variables[0]])
			df[self.X_variables[0]] = pd.to_datetime(df[self.X_variables[0]])
			df = df.set_index(self.X_variables[0])
			df=df.sort_values(by=[self.X_variables[0]],axis=0)
			df=df.reset_index()
			df.columns = ['ds']

			if (self.algorithm == 'ARIMA'):
				predictions=model.predict(start=1,end=df.shape[0]).rename('ARIMA predictions')
				predictions = predictions.tolist()

			else:
				predictions = model.predict(df)
				predictions = predictions['yhat'].tolist()
		else:
			# df = df[0][0]
			x = df[self.X_variables]
			y = df[self.targetvariable]

			predictions = model.predict(x)

		return predictions


	def modeling_performance(self,df,status):

		df.reset_index(inplace = True)
		predictions = self.prediction_func(df)

		output_columns = ['START_DATE','Region','m_sku_desc']

		# print (df[self.targetvariable[0]])
		# print (predictions)

		targets = pd.DataFrame({'Actual_Tons': df[self.targetvariable[0]], 'Predicted_Tons': predictions})
		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding process ID in resultset
		result['ProcessID'] = self.processID

		# adding Name column of used algo
		result ['Algorithm'] = self.algorithm
		
		#Adding accuracy columns in dataframe
		result ['MAPE'] = self.mape(result)
		result ['WMAPE'] = self.wmape(result)
		result ['Error'] = self.error( result['Actual_Tons'], result ['Predicted_Tons'] )

		if (status == 'test'):
			with open('./Modeling_Output/Test_Performance.csv', 'a') as f:
				result.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		else:
			with open('./Modeling_Output/Train_Performance.csv', 'a') as f:
				result.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')

		return result
		

################# This Function moves the created models in
################# the models Repository

	def model_repo(self,result):

		# columns required for Model ID
		repo_cols = ['Region','m_sku_desc','Algorithm','ProcessID']

		# self.Resultant is dataframe having info about all models

		# adding info to repo
		repo = result
		repo = repo[repo_cols]
		repo = repo.drop_duplicates()

		today = date.today()
		d1 = today.strftime("%d/%m/%Y")
		repo['Creation_Date'] = d1

		t = time.localtime()
		current_time = time.strftime("%H:%M:%S", t)
		repo['Creaction_Time'] = current_time

		repo['VariableListID'] = self.variable_list_id
		repo['ParameterID'] = self.parameter_id
		repo['ThresID'] = self.performancethres_id

		#writing models repo in file
		with open('./Modeling_Output/Models_Repo.csv', 'a') as f:
				repo.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

################# This Function saves the performance
################# of each model against its model id

	def evaluate_model(self,result):

		# columns required for Model ID
		comp_cols = ['Region','m_sku_desc','Algorithm','ProcessID','MAPE']

		# self.Resultant is dataframe having info about all models

		# Moving models to completed metadata
		completed = result[comp_cols]
		completed = completed.drop_duplicates()

		completed['VariableListID'] = self.variable_list_id
		completed['ParameterID'] = self.parameter_id
		completed['ThresID'] = self.performancethres_id

		completed.rename(columns={'m_sku_desc': 'SKU', 'Algorithm': 'Models'}, inplace=True)

		with open('./Modeling_Output/Completed_Models.csv', 'a') as f:
				completed.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		errored = completed[(completed['MAPE']>self.errored_thres)]
		with open('./Modeling_Output/Errored_Models.csv', 'a') as f:
				errored.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		active = completed[(completed['MAPE']<self.active_thres)]
		with open('./Modeling_Output/Active_Models.csv', 'a') as f:
				active.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

	def read_scoring_profile (self):
		
		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Scoring"+".csv"
		
		nameofdataset = "./profiles/"+nameofdataset

		forecast_df = pd.read_csv(nameofdataset)

		return forecast_df

	def read_processed_profile (self):
		
		# with open("Prepared_Test_DataSets", 'rb') as t:
		# 	test_df_list = pickle.load(t)
		nameofdataset = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Processed"+".csv"

		nameofdataset = "./profiles/"+nameofdataset

		forecast_df = pd.read_csv(nameofdataset)

		return forecast_df			



	def compare_profiles(self,p_data,s_data):

		df_colnames_list=['Median','Unique_values','Mean','Minimum','Maximum','Quantile_25','Quantile_75','Std_Dev','Null_Count']

		a=pd.DataFrame(p_data)

		b=pd.DataFrame(s_data)

		b=b[df_colnames_list]

		b=b.rename(columns={'Median':'MedianS','Unique_values':'Unique_valuesS','Mean':'MeanS','Minimum':'MinimumS','Maximum':'MaximumS','Quantile_25':'Quantile_25S','Quantile_75':'Quantile_75S','Std_Dev':'Std_DevS','Null_Count':'Null_CountS'})

		a.reset_index(drop=True, inplace=True)

		b.reset_index(drop=True, inplace=True)

		dt=pd.concat([a,b],axis=1)

        
		#Finding Maximum Values 

		median=dt[['Median','MedianS']].max(axis=1)

		Unique_values=dt[['Unique_values','Unique_valuesS']].max(axis=1)

		mean=dt[['Mean','MeanS']].max(axis=1)

		minimum=dt[['Minimum','MinimumS']].max(axis=1)

		maximum=dt[['Maximum','MaximumS']].max(axis=1)

		quantile25=dt[['Quantile_25','Quantile_25S']].max(axis=1)

		quantile75=dt[['Quantile_75','Quantile_75S']].max(axis=1)

		std_dev=dt[['Std_Dev','Std_DevS']].max(axis=1)

		mode=dt[['Null_Count','Null_CountS']].max(axis=1)


		#Finding Percentage Difference

		c1=round(abs(((a['Median']-b['MedianS'])/median)*100),2)

		c2=round(abs(((a['Unique_values']-b['Unique_valuesS'])/Unique_values)*100),2)

		c3=round(abs(((a['Mean']-b['MeanS'])/mean)*100),2)

		c4=round(abs(((a['Minimum']-b['MinimumS'])/minimum)*100),2)

		c5=round(abs(((a['Maximum']-b['MaximumS'])/maximum)*100),2)

		c6=round(abs(((a['Quantile_25']-b['Quantile_25S'])/quantile25)*100),2)

		c7=round(abs(((a['Quantile_75']-b['Quantile_75S'])/quantile75)*100),2)

		c8=round(abs(((a['Std_Dev']-b['Std_DevS'])/std_dev)*100),2)

		c9=round(abs(((a['Null_Count']-b['Null_CountS'])/mode)*100),2)

		#Making one last Dataframe 

		df=pd.DataFrame(zip(c1,c2,c3,c4,c5,c6,c7,c8,c9),columns=df_colnames_list)

		#Remove NaNs And Inf to 0

		df.fillna(0, inplace=True)

		df.replace(np.inf,0,inplace=True)

		#Making csv file out of dataframe

		os.makedirs('./Comparison_Output', exist_ok = True)

		nameofprofile = "PID_"+str(self.processID)+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_"+"Comparsion"+".csv"

		nameofprofile = "./Comparison_Output/"+nameofprofile

		with open(nameofprofile, 'w') as f:

				df.to_csv(f, index = False,  mode='w', header=f.tell()==0, line_terminator='\n')


	def check_comparison(self):
		
		path = r"./Comparison_Output/*.csv"

		#Getting all files from Comparison_Output folder

		for fname in glob.glob(path):

			print(fname)

			df=pd.read_csv(fname)

			col_names=[x for x in df.columns]

			z=0

		#Finding all values in each column greater than 40	

			for x in col_names:

				for y in df[x]:

					if(y>50):

						z+=1
				if(z>=40):

					print('True in',x)

				else:

					print('False in',x)

				z=0

	def maintain_inprogress_process(self,processid):

		df=pd.read_csv('Processes_Table.csv')

		x=df.loc[(df['ProcessID']==processid)]

		x.to_csv('In_progress.csv',index=False)


	def maintain_completed_process(self,processid):

		df=pd.read_csv('In_progress.csv')

		index_names = df[df['ProcessID'] == processid ].index 

		df2=pd.DataFrame(df[df['ProcessID'] == processid ])

		file_name = "Completed.csv"

		if os.path.exists(file_name):

			# don't include headers if the file already exists
			df2.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
		else:

			# include headers if the file does not exist
			df2.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')

		# df2.to_csv('Completed.csv',index=False, mode='a', header=True)

		df.drop(index_names, inplace = True)

		df.to_csv('In_progress.csv',index=False)

##############################################################################################################

					####### Scoring Functionalities #######

################# This is the orchestration function of Scoring
################# which call all the Scoring functionalities in order

	def scoring_orchestrator(self):


		os.makedirs('./Scoring_Output', exist_ok = True)
		# reading models/algos from log and saving in self.algos
		log = pd.read_csv(config.Active_Models)
		var_table = pd.read_csv(config.variable_table)
		log.reset_index(inplace = True)
		totalrows = log.shape[0]


		i=0

		while i<totalrows:

			self.read_model_backlog(log,i)

			forecastdata = self.read_forecastdata()

			forecastdata = self.Data_Prep(forecastdata,var_table)		

			self.data_profile(forecastdata, "Scoring")
			p_data=self.read_processed_profile()
			s_data=self.read_scoring_profile()

			self.compare_profiles(p_data,s_data)
			self.check_comparison()

			predictions = self.prediction_func(forecastdata)

			self.save_predictions(forecastdata,predictions)

			df = self.scoring_performance(forecastdata,i).Resultset
			#result = self.scoring_performance(testdata[0][0],predictions)

			i=i+1

		self.rank_performance(df,'./Scoring_Output/Resultant.csv')


################# This Function saves the predictions 
################# for each model

	def save_predictions(self,df,predictions):

		df.reset_index(inplace = True)

		output_columns = ['START_DATE','Region','m_sku_desc']

		targets = pd.DataFrame({'Predicted_Tons': predictions})

		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding Name column of used algo
		result ['Algorithm'] = self.algorithm

		# adding process ID in resultset
		result['ProcessID'] = self.processID


		with open('./Scoring_Output/FurutePredictions.csv', 'a') as f:
				result.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')





################# This Function calculates the accuracy of predictions
################# my the models

	def scoring_performance(self,df,i):

		predictions = self.prediction_func(df)

		output_columns = ['START_DATE','Region','m_sku_desc']


		targets = pd.DataFrame({'Actual_Tons': df[self.targetvariable[0]], 'Predicted_Tons': predictions})
		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding process ID in resultset
		result['ProcessID'] = self.processID

		# adding Name column of used algo
		result ['Algorithm'] = self.algorithm
		
		#Adding accuracy columns in dataframe
		result ['MAPE'] = self.mape(result)
		result ['WMAPE'] = self.wmape(result)
		result ['Error'] = self.error( result['Actual_Tons'], result ['Predicted_Tons'] )


		if i==0:
			self.Resultset = result
		else:
			self.Resultset=pd.concat([self.Resultset, result],axis=0,sort=False)

		
		# with open('Train_Performance.csv', 'a') as f:
		# 		result.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')
		return self


################# This Function ranks the models as per their performance
################# in each sku and region and selects champion model in that sku and region



	def rank_performance(self,df,nameoffile):

		# columns needed in output
		comp_cols = ['Region','m_sku_desc','Algorithm','ProcessID','MAPE']
		self.Resultset = df
		# self.Resultant has the results of predictions and accuracies
		data = self.Resultset[comp_cols]

		# extracting unique models with their MAPE
		data = data.drop_duplicates()

		# Extracting unique regions and skus
		region_list = data.Region.unique()
		SKU_list = data.m_sku_desc.unique()

		# creating a list to separate the results of each region,sku
		frames = [None for i in range(len(region_list)*len(SKU_list))]

		#creating an empty dataset to save final result with ranks
		rank = data.copy()
		rank.drop(rank.index, inplace=True)

		i = 0

		# Looping through each region,sku
		for reg in region_list:
			for sku in SKU_list:

				# saving results of models one region,sku in frames[i]
				frames[i] = data[(data['Region'] == reg) & (data['m_sku_desc'] == sku)]
				# sorting all the models according to MAPE
				frames[i] = frames[i].sort_values(by=['MAPE'], ascending=True)
				# Assigning them ranks from top to bottom
				frames[i]['Rank'] = range(1,frames[i].shape[0]+1,1)
				# creating an empty champion column
				frames[i]['Champion'] = None

				#Assigning champion to the model in first row of this resultset
				frames[i].reset_index(inplace=True)
				frames[i]['Champion'][0] = 'Champion'
				#removing useless column
				frames[i].drop('index',axis = 1, inplace = True)
				#joining results of this region,sku with final resultset
				rank = pd.concat([rank,frames[i]], axis=0, ignore_index=True)



				i = i+1

		# changing string rank to int
		rank = rank.astype({"Rank":int})

		#joining the Rank resultset with predictions resultset
		resultset = pd.merge(self.Resultset, rank, on=['Algorithm','ProcessID','Region','m_sku_desc','MAPE'], how='left')

		# Saving final results in DB
		resultset.to_csv(nameoffile,index = False)



################# This Function Reads the log as an argument from metadata
################# Also, it saves the profile of variables in scoring data

	def score_data_profile(self, log):

		# Open test datasets file
		with open("Prepared_Test_DataSets", 'rb') as t:
			self.test_df_list = pickle.load(t)


		# Save region counts
		self.rcount = len(self.test_df_list)

		# Save skus count
		self.skucount = len(self.test_df_list[0])

		# Get the last row number
		last_row_number = log.shape[0]



		# Get the last row
		last_row = log.loc[[last_row_number - 1]]



		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']



		# Importing Data From File
		data = pd.read_csv("Variable_Table.csv")

		# Import variable table in dataframe
		variable_Table = pd.read_csv('Variable_Table.csv')

		# Select reduced variables
		self.reduced_Var = variable_Table[(variable_Table.Influencer_Cat == "Media") | (variable_Table.Influencer_Cat == "Price") | (variable_Table.Influencer_Cat == "Incentive") ]['Variables'].tolist()


		# Iterate through the datframes
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:


				# Select reduced variables only
				df = self.test_df_list[i][j][self.reduced_Var]



				# save data profiles in variables
				prof_median = df.median(numeric_only=bool)
				prof_unique = df.nunique(dropna=True)
				prof_mean = df.mean(numeric_only=bool)
				prof_mode = df.mode(numeric_only=bool)
				prof_min = df.min(numeric_only=bool)
				prof_max = df.max(numeric_only=bool)
				prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
				prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
				prof_std = df.std(numeric_only=bool)
				prof_nan_count = len(df) - df.count()




				# create a dataframe to be exported
				result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)



				# reset index
				result.reset_index(level = 0, inplace =True)



				# rename columns
				result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)



				# insert current date into the column
				result['Date'] = str(date.today())



				# insert current time into the column
				result['Time'] = str(datetime.now().time())



				# insert backlog id into column
				result['ProcessID'] = ProcessID



				# insert Data Stage
				result['Data_Stage'] = 'Scoring'



				# insert mode column as NULL (For now)
				result['Mode'] = ''



				# insert region column
				test = self.test_df_list[i][j].Region.unique()
				print(test[0])
				result['Region'] = test[0]



				# insert sku column
				test1 = self.test_df_list[i][j].m_sku_desc.unique()
				result['Sku'] =test1[0]



				# Name of the file we want to write to
				file_name = "Profile_Table.csv"



				# Exporting the file
				# Check the availability of the file
				if os.path.exists(file_name):



					# don't include headers if the file already exists
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
				else:



					# include headers if the file does not exist
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')




				j += 1
			i += 1
