#import modules you'll have to pip install missing ones
from pathlib import Path #Path module for filepath handling
import argparse #command line argument parsing
import pandas as pd #data handling lib
import numpy as np #maths lib
from sklearn.linear_model import LinearRegression, Ridge #regression model lib, grabbing specifically the 2 models we want to use
from sklearn.metrics import r2_score #r2 is a regression "score" metric
import matplotlib.pyplot as plt #plotter lib

#Set respective files to their apt names here, top 2 are the system indices, the bottom 3 are the 3 weather stations used
AAOI_FNAME = "aaoi.csv"
MEI_FNAME = "mei.csv"
ANTO_FNAME = "AntoWS.csv"
PUERTO_FNAME = "PuertoWS.csv"
QUIN_FNAME = "QuinWS.csv"

###
#Data processing area to read and load our data into appropriate frames or series
###
def load_data(base):
	meipath = base / MEI_FNAME #For the mei file path
	meivalues = [] #make an empty list to store all the mei values
	with meipath.open("r", encoding="utf-8") as f: #open the mei file to be read
		for line in f: #for every line in the mei file we run this loop
			s = line.strip() #remove the blank spaces and newline characters
			if not s or s.startswith("#"): #skip blanks or hased lines
				continue #next line
			meivalues.append(float(s)) #appends that line as a float to our meivalues list
	mei_idx = pd.date_range(start="1979-01-01", periods=len(meivalues), freq="MS") #starts a date range for the mei values as the file is missing them, starting jan 1979, monthly frequency
	mei = pd.Series(data=meivalues, index=mei_idx, name="MEI") #turns our mei values list into a series with the date index using pandas

	aaoipath = base / AAOI_FNAME #much easier setup for aaoi as it has dates attacthed already
	aaoi = pd.read_csv(aaoipath, header=None, names=["date", "value"], parse_dates=[0], index_col=0) #read the csv and set up a date and value column, parse the dates, and set the date column as the index
	aaoi["value"] = aaoi["value"].astype(str).str.replace("D", "E").astype(float) #Convert to D style notation to E style for floats
	aaoi = aaoi["value"].rename("AAOI") #set the series name to AAOI
	#station processing
	stations = {} #dict to hold the station dataframes
	for fname in [ANTO_FNAME, PUERTO_FNAME, QUIN_FNAME]: #for each of our station files
		p = base / fname #path to the file
		df = pd.read_csv(p, parse_dates=[0]) #read it and parse the date column
		if "time" in df.columns: #if time is a column
			df = df.set_index("time") #make the time column the index
		df["t2m_C"] = df["t2m"] - 273.15 #convert our temperature to celcius 
		df["wind_ms"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2) #turn our wind components into a single wind speed 
		stations[fname.replace('.csv', '')] = df[["t2m_C", "wind_ms"]].copy() #store the columns we want in the dict, remove the .csv too

	return {"mei": mei, "aaoi": aaoi, "stations": stations} #return the final dict of data for our use

###
#Regression table setup area to prep data for regression
###
def reg_tables(mei, aaoi, stations):
	base_df = pd.concat([mei, aaoi], axis=1) #make a base dataframe with our 2 indices 

	combined = [] #empty list for station dataframes
	for name, df in stations.items(): #for each station in our stations dict we loop this
		df2 = df.rename(columns={"t2m_C": f"t2m_{name}", "wind_ms": f"wind_{name}"}) #suffix the columns with the appropriate station name to keep them seperate
		combined.append(df2) #then append to the empty list
	combined_df = pd.concat(combined, axis=1) #pandas concat makes a big dataframe with everything together 
	t2m_cols = [c for c in combined_df.columns if c.startswith("t2m_")] #select all the temperature columns 
	wind_cols = [c for c in combined_df.columns if c.startswith("wind_")] #select all our wind columns too
	combined_df["regional_t2m_mean"] = combined_df[t2m_cols].mean(axis=1) #new column that means all temperature columns
	combined_df["regional_wind_mean"] = combined_df[wind_cols].mean(axis=1) #and a new column that means all wind columns

	merged = base_df.join(combined_df[["regional_t2m_mean", "regional_wind_mean"]], how="inner") #merge the base dataframe with the regional means

	per_station = {} #empty dict for per station dataframes again
	for name in stations.keys(): #for each station in our station dict
		tcol = f"t2m_{name}" #set the temp column name to the appropriate station
		wcol = f"wind_{name}" #same for wind
		df = merged[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].join(combined_df[[tcol, wcol]], how="left") #make a dataframe with the indices, regional means, and the station specific columns
		df = df.rename(columns={tcol: "target", wcol: "weather_station"}) #rename the station specific columns to target and weather_station for regression use
		per_station[name] = df #store this in our new station dict
	return per_station #return the final dict of per station dataframes

###
#Reporting area for regression results and plots
###
def reporting(per_station, outputdir, use_ridge=False):
	results = [] #empty list for results
	datesplit = pd.Timestamp("2000-01-01") #date to split pre and post 2000
	outputdir.mkdir(parents=True, exist_ok=True) #make output directory if it doesnt exist
	plotoutput = outputdir / "plots" #sets plot output dir
	plotoutput.mkdir(parents=True, exist_ok=True) #make plot output dir if it doesnt exist

	# get reporting mode and models from arguments set on run
	mode = getattr(reporting, "mode", "within") #default to within if not set
	models_to_run = getattr(reporting, "models", ("OLS", "Ridge")) #model types to run
	model_map = {"OLS": LinearRegression, "Ridge": Ridge} #map models to classes

	for name, df in per_station.items(): #for each station in our per station dict
		df = df.dropna(subset=["target", "MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]).copy() #drop rows with nans in important columns
		pre = df[df.index < datesplit] #split into pre and post 2000 dataframes
		post = df[df.index >= datesplit] #post 2000 dataframe

		#validation mode is basically training on pre2000 data to test on post 2000 data
		if mode == "validate": #if set to validate
			train_df, test_df = pre, post #train on pre2000, test on post2000
			train_name, test_name = "pre2000", "post2000" #sets names

			X_train = train_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x value training set
			y_train = train_df["target"].values #target y training values
			X_test = test_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x value test set
			y_test = test_df["target"].values #target y test values

			for model_name in models_to_run: #for each model to run
				ModelCls = model_map.get(model_name) #set model class
				model = ModelCls() #state model
				model.fit(X_train, y_train) #fit model to training data
				ypred_train = model.predict(X_train) #predict on training data
				ypred_test = model.predict(X_test) #predict on test data
				r2_train = float(r2_score(y_train, ypred_train)) if len(y_train) > 0 else float("nan") #r2 score for training data
				r2_test = float(r2_score(y_test, ypred_test)) if len(y_test) > 0 else float("nan") #r2 score for test data
				coefs = list(model.coef_) #list our coefficients
				intercept = float(model.intercept_) #get the intercept

				results.append({ #append our results
					"station": name, #respective station name
					"period": f"train_{train_name}_test_{test_name}", #period info (pre or post 2000, train or test data)
					"model": model_name, #used model name
					"n_rows_train": train_df.shape[0], #number of rows in training data
					"n_rows_test": test_df.shape[0], #number of rows in test data
					"intercept": intercept, #the intercept value
					"coef_MEI": coefs[0] if len(coefs) > 0 else float("nan"), #coefficient for MEI
					"coef_AAOI": coefs[1] if len(coefs) > 1 else float("nan"), #coefficient for AAOI
					"coef_reg_t2m": coefs[2] if len(coefs) > 2 else float("nan"), #coefficient for regional t2m mean
					"coef_reg_wind": coefs[3] if len(coefs) > 3 else float("nan"), #coefficient for regional wind mean
					"R2_train": r2_train, #R2 score for training data
					"R2_test": r2_test, #R2 score for test data
				})

				rpt = outputdir / f"report_{name}_train_{train_name}_test_{test_name}_{model_name}.txt" #set a file path for the end report
				with rpt.open("w", encoding="utf-8") as f: #open the report file
					f.write(f"Station: {name}\nMode: validate\nTrain: {train_name}\nTest: {test_name}\nModel: {model_name}\nRows (train/test): {train_df.shape[0]} / {test_df.shape[0]}\n\n") #write's all the aformentioned info into the csv to be read as a report file
					f.write(f"Intercept: {intercept}\n")
					f.write(f"coef_MEI: {coefs[0] if len(coefs) > 0 else 'nan'}\n")
					f.write(f"coef_AAOI: {coefs[1] if len(coefs) > 1 else 'nan'}\n")
					f.write(f"coef_reg_t2m: {coefs[2] if len(coefs) > 2 else 'nan'}\n")
					f.write(f"coef_reg_wind: {coefs[3] if len(coefs) > 3 else 'nan'}\n")
					f.write(f"R2_train: {r2_train}\nR2_test: {r2_test}\n")

				# scatter for test set (observed vs predicted)
				fig, ax = plt.subplots(figsize=(6, 6)) #sets figure and axis and size (6x6)
				if len(y_test) > 0: #if there are test y values
					ax.scatter(y_test, ypred_test, s=20, alpha=0.7) #scatter plot of observed vs predicted, size of plots and alpha is transperency
					mn = min(y_test.min(), ypred_test.min()) #min value for 1:1 line
					mx = max(y_test.max(), ypred_test.max()) #max val
					ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="1:1") #makes a 1:1 line (basically a perfect prediction line)
				ax.set_xlabel("Observed 2m Surface Temperature, C") #x axis label
				ax.set_ylabel("Predicted 2m Surface Temperature, C") #y axis label
				ax.set_title(f"{name} validate {model_name} Test Observed vs Predicted") #title
				ax.legend() #legend, we should probs add more to this
				fig.tight_layout() #tight layout so things dont overlap
				scatter_path = plotoutput / f"{name}_validate_{model_name}_obs_vs_pred.png" #path to save to and filename to save to
				fig.savefig(scatter_path) #saves it
				plt.close(fig) #closes the file

				#time series graph
				fig, ax = plt.subplots(figsize=(10, 4)) #figure and axis with size 10x4
				if len(test_df) > 0: #if there is test data 
					ax.plot(test_df.index, y_test, label="observed") #plot observed values with
					ax.plot(test_df.index, ypred_test, label="predicted") #predicted values 
				ax.set_ylabel("2m Surface Temperature, C") #y axis label
				ax.set_title(f"{name} validate {model_name} Test Time Series") #title
				ax.legend() #legend, same as before im sure we could add more than just the 1:1 line
				fig.tight_layout() #tight layout so things dont overlap
				ts_path = plotoutput / f"{name}_validate_{model_name}_timeseries.png" #path to save to and filename to save to
				fig.savefig(ts_path) #saves it
				plt.close(fig) #closes the file

		else:
			#non-validated plotting area
			for period_name, period_df in (("pre2000", pre), ("post2000", post)): #for each period (pre and post 2000) we loop this
				X = period_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x values for the period
				y = period_df["target"].values #target y values for the period

				for model_name in models_to_run: #for each model to run
					ModelCls = model_map.get(model_name) #gets the model class
					model = ModelCls() #class initialisation
					model.fit(X, y) #fit the model to the data
					ypred = model.predict(X) #predict y values
					r2 = float(r2_score(y, ypred)) if len(y) > 0 else float("nan") #r2 score calculation
					coefs = list(model.coef_) #list of coefficients
					intercept = float(model.intercept_) #intercept value
					results.append({ #same as before, appending results to the list
						"station": name,
						"period": period_name,
						"model": model_name,
						"n_rows": period_df.shape[0],
						"intercept": intercept,
						"coef_MEI": coefs[0] if len(coefs) > 0 else float("nan"),
						"coef_AAOI": coefs[1] if len(coefs) > 1 else float("nan"),
						"coef_reg_t2m": coefs[2] if len(coefs) > 2 else float("nan"),
						"coef_reg_wind": coefs[3] if len(coefs) > 3 else float("nan"),
						"R2": r2,
					})

					rpt = outputdir / f"report_{name}_{period_name}_{model_name}.txt" #again same as before makes a report file, but for non-validated data
					with rpt.open("w", encoding="utf-8") as f:
						f.write(f"Station: {name}\nPeriod: {period_name}\nModel: {model_name}\nRows: {period_df.shape[0]}\n\n")
						f.write(f"Intercept: {intercept}\n")
						f.write(f"coef_MEI: {coefs[0] if len(coefs) > 0 else 'nan'}\n")
						f.write(f"coef_AAOI: {coefs[1] if len(coefs) > 1 else 'nan'}\n")
						f.write(f"coef_reg_t2m: {coefs[2] if len(coefs) > 2 else 'nan'}\n")
						f.write(f"coef_reg_wind: {coefs[3] if len(coefs) > 3 else 'nan'}\n")
						f.write(f"R2: {r2}\n")

					#further plotting area for non-validated data
					fig, ax = plt.subplots(figsize=(6, 6)) #these plots are the same as above but they just use the non-validated data
					ax.scatter(y, ypred, s=20, alpha=0.7)
					if len(y) > 0:
						mn = min(y.min(), ypred.min())
						mx = max(y.max(), ypred.max())
						ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="1:1")
					ax.set_xlabel("Observed 2m Surface Temperature, C")
					ax.set_ylabel("Predicted 2m Surface Temperature, C")
					ax.set_title(f"{name} {period_name} {model_name} Observed vs Predicted")
					ax.legend()
					fig.tight_layout()
					scatter_path = plotoutput / f"{name}_{period_name}_{model_name}_obs_vs_pred.png"
					fig.savefig(scatter_path)
					plt.close(fig)

					fig, ax = plt.subplots(figsize=(10, 4))
					ax.plot(period_df.index, y, label="observed")
					ax.plot(period_df.index, ypred, label="predicted")
					ax.set_ylabel("2m Surface Temperature, C")
					ax.set_title(f"{name} {period_name} {model_name} Time Series")
					ax.legend()
					fig.tight_layout()
					ts_path = plotoutput / f"{name}_{period_name}_{model_name}_timeseries.png"
					fig.savefig(ts_path)
					plt.close(fig)

	outputdf = pd.DataFrame(results) #final dataframe from results list
	outputdf.to_csv(outputdir / "regression_summary.csv", index=False) #writes the dataframe to a csv file
	print("Wrote regression summary to", outputdir / "regression_summary.csv") #prints out where the csv was written
	print("Wrote plots to", outputdir / "plots") #print out where the plots were written

###
#Function area for our argument parsing
###
def main():
	parser = argparse.ArgumentParser() #sets up the argument parser
	parser.add_argument("--run", action="store_true", help="actual regression run, could just default true") #run argument to actually run the regression
	parser.add_argument("--dir", "-d", default=None, help="will just default to wkdir") #directory argument to set the working directory, defaults to current script dir
	parser.add_argument("--validate", action="store_true", help="Train on pre2000 and validate on post2000") #validate argument to train on pre2000 and validate on post2000
	parser.add_argument("--models", choices=("both", "OLS", "Ridge"), default="both", #OLS is ordinary least squares, its the "normal" for linear regression with multiple variables
						help="model run, OLS is ordinary least squares")
	args = parser.parse_args() #parses the arguments

	base = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent #sets the base directory to either the arg provided or the current script directory

	if not args.run: #if nothings used drop this messge, we could just default to true
		print("use the shown args (--run) to properly start") 
		return 

	data = load_data(base) #loads the data from the base directory
	pstation = reg_tables(data["mei"], data["aaoi"], data["stations"]) #prepares the regression tables
	outputdirec = base / "regression_out" #output directory for regression results

	if args.validate: #if validate argument is used
		reporting.mode = "validate" #set reporting mode to validate (train pre2000, test post2000)
	else:
		reporting.mode = "within" #set reporting mode to just the whole period

	if args.models == "both": #if both models argument is used
		reporting.models = ("OLS", "Ridge") #set reporting models to both
	else: #otherwise
		reporting.models = (args.models,) #set reporting models to the single model provided

	reporting(pstation, outputdirec) #run the reporting function with the per station data and output directory


if __name__ == "__main__":
	main() #entry point for the script, calls main function, kinda weird python standard

