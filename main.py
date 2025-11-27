
#
#Arguments at the bottom, 
#Parts are a bit bloated for clarity I feel it could be made much more efficient but this works fine having all our plots come out right now
#

#import modules you'll have to pip install missing ones
from pathlib import Path #Path module for filepath handling
import argparse #command line argument parsing
import pandas as pd #data handling lib
import numpy as np #maths lib
from sklearn.linear_model import LinearRegression, Ridge #regression model lib, grabbing specifically the 2 models we want to use
from sklearn.metrics import r2_score #r2 is a regression "score" metric
import matplotlib.pyplot as plt #plotter lib
from sklearn.pipeline import make_pipeline #so we can scale into regression
from sklearn.preprocessing import StandardScaler #scaling for ridge regression

#Set respective files to their apt names here, top 2 are the system indices, the bottom 3 are the 3 weather stations used
AAOI_FNAME = "aaoi.csv"
MEI_FNAME = "mei.csv"
ANTO_FNAME = "AntoWS.csv"
PUERTO_FNAME = "PuertoWS.csv"
QUIN_FNAME = "QuinWS.csv"

#For the graphs so the actual names of the places are used
STATION_ACTUALS = {
	"AntoWS": "Antofagasta",
	"PuertoWS": "Puerto Montt el Tepual",
	"QuinWS": "Quintero",
}


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
	# (we preserve the attribute approach but this function now performs validation-only)
	models_to_run = getattr(reporting, "models", ("OLS", "Ridge")) #model types to run
	model_map = {"OLS": LinearRegression, "Ridge": Ridge} #map models to classes

	for name, df in per_station.items(): #for each station in our per station dict
		df = df.dropna(subset=["target", "MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]).copy() #drop rows with nans in important columns
		pre = df[df.index < datesplit] #split into pre and post 2000 dataframes
		post = df[df.index >= datesplit] #post 2000 dataframe

		# validation: train on pre2000, test on post2000
		train_df, test_df = pre, post #train on pre2000, test on post2000
		train_name, test_name = "pre2000", "post2000" #sets names

		X_train = train_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x value training set
		y_train = train_df["target"].values #target y training values
		X_test = test_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x value test set
		y_test = test_df["target"].values #target y test values

		# collect model predictions so we can optionally produce a 2x2 panel combining OLS and Ridge
		model_preds = {}

		for model_name in models_to_run: #for each model to run
			ModelCls = model_map.get(model_name) #set model class
			if ModelCls is None:
				continue
			if model_name == "Ridge": #if ridge is selected then standardise the data first
				model = make_pipeline(StandardScaler(), Ridge()) #using a pipeline from sklearn we can do this, standardising is to mitigate sensitivity to variable scaling
			else: 
				model = ModelCls()

			model.fit(X_train, y_train) #fit model to training data
			ypred_train = model.predict(X_train) #predict on training data
			ypred_test = model.predict(X_test) #predict on test data
			r2_train = float(r2_score(y_train, ypred_train)) if len(y_train) > 0 else float("nan") #r2 score for training data
			r2_test = float(r2_score(y_test, ypred_test)) if len(y_test) > 0 else float("nan") #r2 score for test data
			model_preds[model_name] = {"ypred_test": ypred_test, "r2_test": r2_test}
			final_est = model #default final estimator is the model itself
			if hasattr(model, "named_steps"): #check for named_steps (its a pipeline attribute)
				if "ridge" in model.named_steps: #check for ridge step
					final_est = model.named_steps["ridge"] #set final estimator to ridge step
			coefs = list(final_est.coef_) if hasattr(final_est, "coef_") else [float("nan")] * 4 #get all our coefficients
			intercept = float(final_est.intercept_) if hasattr(final_est, "intercept_") else float("nan") #get the intercept

			#Statistical test/diagnostics stuff to see how the models did
			rmse = float("nan") #root mean squared error, this is a regression error metric to see how well the model did
			mae = float("nan") #mean absolute error, another regression error metric
			base_rmse = float("nan") #given a baseline rmse value to create skill score 
			skillsc = float("nan") #skill score to see how well the model did against climatology defaults
			if len(y_test) > 0: #loop for if there are test variables present
				diffs = y_test - ypred_test #differences between observed and predicted
				rmse = float(np.sqrt(np.mean(diffs ** 2))) #calculate rmse this is the square root of the mean of the squared differences
				mae = float(np.mean(np.abs(diffs))) #calculate mae, this is the mean of the absolute differences
				trainmean = float(np.mean(y_train)) if len(y_train) > 0 else float("nan") #mean of training y values
				baselines = np.full_like(y_test, trainmean, dtype=float) #baseline predictions as the train mean
				base_rmse = float(np.sqrt(np.mean((y_test - baselines) ** 2))) #baseline rmse calculation
				if base_rmse and not np.isnan(base_rmse): #if baseline rmse is valid
					skillsc = 1.0 - (rmse / base_rmse) #set skill score to 1- the rmse output over the baseline rmse
			if model_name in model_preds: #to store the rmse to be used on our graphs
				model_preds[model_name]["rmse_test"] = rmse #puts the rmse output in a dict for use later

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
				"RMSE_test": rmse, #root mean squared error on test set
				"MAE_test": mae, #mean absolute error on test set
				"baseline_rmse": base_rmse, #RMSE of climatology baseline (train mean)
				"skill_score": skillsc, #1 - RMSE_model / RMSE_climatology
			})

			# write per-station report
			rpt = outputdir / f"report_{name}_train_{train_name}_test_{test_name}_{model_name}.txt" #set a file path for the end report
			with rpt.open("w", encoding="utf-8") as f: #open the report file
				f.write(f"Station: {name}\nMode: validate (train pre2000 -> test post2000)\nModel: {model_name}\nRows (train/test): {train_df.shape[0]} / {test_df.shape[0]}\n\n") #write's all the aformentioned info into the csv to be read as a report file
				f.write(f"Intercept: {intercept}\n")
				f.write(f"coef_MEI: {coefs[0] if len(coefs) > 0 else 'nan'}\n")
				f.write(f"coef_AAOI: {coefs[1] if len(coefs) > 1 else 'nan'}\n")
				f.write(f"coef_reg_t2m: {coefs[2] if len(coefs) > 2 else 'nan'}\n")
				f.write(f"coef_reg_wind: {coefs[3] if len(coefs) > 3 else 'nan'}\n")
				f.write(f"R2_train: {r2_train}\nR2_test: {r2_test}\n")
				f.write(f"RMSE_test: {rmse}\nMAE_test: {mae}\n")
				f.write(f"baseline_rmse (climatology): {base_rmse}\n")
				f.write(f"skill_score (1 - RMSE_model/RMSE_climatology): {skillsc}\n")

			# plots
			fig, ax = plt.subplots(figsize=(6, 6)) #sets figure and axis and size (6x6)
			if len(y_test) > 0: #if there are test y values
				ax.scatter(y_test, ypred_test, s=20, alpha=0.7) #scatter plot of observed vs predicted
				mn = min(y_test.min(), ypred_test.min()) #min value for 1:1 line
				mx = max(y_test.max(), ypred_test.max()) #max val
				ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="1:1") #1:1 line
			ax.set_xlabel("Observed 2m Surface Temperature, C") #x axis label
			ax.set_ylabel("Predicted 2m Surface Temperature, C") #y axis label
			ax.set_title(f"{name} validate {model_name} Test Observed vs Predicted") #title
			# annotate R2 values on the scatter plot
			r2_train_txt = f"R2_train: {r2_train:.3f}" if not np.isnan(r2_train) else "R2_train: nan"
			r2_test_txt = f"R2_test: {r2_test:.3f}" if not np.isnan(r2_test) else "R2_test: nan"
			ax.text(0.02, 0.98, r2_train_txt + "\n" + r2_test_txt, transform=ax.transAxes,
				fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
			ax.legend()
			fig.tight_layout()
			scatter_path = plotoutput / f"{name}_validate_{model_name}_obs_vs_pred.png" #path to save scatter
			fig.savefig(scatter_path)
			plt.close(fig)

			#time series graph
			fig, ax = plt.subplots(figsize=(10, 4)) #figure and axis with size 10x4
			if len(test_df) > 0: #if there is test data
				ax.plot(test_df.index, y_test, label="observed") #plot observed values
				ax.plot(test_df.index, ypred_test, label="predicted") #predicted values
			ax.set_ylabel("2m Surface Temperature, C") #y axis label
			ax.set_title(f"{name} validate {model_name} Test Time Series") #title
			# annotate R2 for timeseries plots (test R2 is most relevant for plotted test series)
			ax.text(0.02, 0.98, r2_test_txt, transform=ax.transAxes, fontsize=9, va="top",
				bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
			ax.legend()
			fig.tight_layout()
			ts_path = plotoutput / f"{name}_validate_{model_name}_timeseries.png" #path to save timeseries
			fig.savefig(ts_path)
			plt.close(fig)

		#When both models are used we can do an output that side by sides them
		if set(model_preds.keys()) >= {"OLS", "Ridge"}: #So a generic if statement for when they're both selected
			ypred_ols = model_preds["OLS"]["ypred_test"] #predicted values for OLS
			ypred_ridge = model_preds["Ridge"]["ypred_test"] #predicted valyues for ridge
			r2_ols = model_preds["OLS"].get("r2_test", float("nan")) #gets the r2 value for ols
			r2_ridge = model_preds["Ridge"].get("r2_test", float("nan")) #gets the r2 value for ridge
			rmse_ols = model_preds["OLS"].get("rmse_test", float("nan")) #gets the rmse value for ols
			rmse_ridge = model_preds["Ridge"].get("rmse_test", float("nan")) #get the rmse value for ridge

			mn = min(y_test.min(), ypred_ols.min(), ypred_ridge.min()) #min and max values for the axis limits
			mx = max(y_test.max(), ypred_ols.max(), ypred_ridge.max())

			display_name = STATION_ACTUALS.get(name, name.replace('WS', '').replace('_', ' ')) #Sets the name to the mapped one in the top dict
			fig, axes = plt.subplots(1, 2, figsize=(12, 5)) #sets up a subplot in a 1x2 format (and gives our figure size)
			ax0, ax1 = axes[0], axes[1] #sets the axes for the plots

			#OLS plot, it's a scatter of observed vs predicted with a 1:1 line, we also add the RMSE and R2 values on it 	
			ax0.scatter(y_test, ypred_ols, s=20, alpha=0.7, color="C0") #x vs y, further paramaters are point settings
			ax0.plot([mn, mx], [mn, mx], color="k", linestyle="--") #this is our 1:1 line to show a perfect 1:1 prediction to observed
			ax0.set_title(f"OLS: Observed vs Predicted — {display_name}") #title
			ax0.set_xlabel("Observed 2m Surface Temperature, C") #x axis
			ax0.set_ylabel("Predicted 2m Surface Temperature, C") #y axis
			ax0.text(0.02, 0.98, f"R2: {r2_ols:.3f}\nRMSE: {rmse_ols:.3f}" if not np.isnan(r2_ols) else f"RMSE: {rmse_ols:.3f}", #our added annotations
				transform=ax0.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
			
			#Then the same thing as above but with the Ridge model
			ax1.scatter(y_test, ypred_ridge, s=20, alpha=0.7, color="C1")
			ax1.plot([mn, mx], [mn, mx], color="k", linestyle="--")
			ax1.set_title(f"Ridge: Observed vs Predicted — {display_name}")
			ax1.set_xlabel("Observed 2m Surface Temperature, C")
			ax1.set_ylabel("Predicted 2m Surface Temperature, C")
			ax1.text(0.02, 0.98, f"R2: {r2_ridge:.3f}\nRMSE: {rmse_ridge:.3f}" if not np.isnan(r2_ridge) else f"RMSE: {rmse_ridge:.3f}",
				transform=ax1.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

			fig.tight_layout() #Sets both of their layouts and saves them to our plot output area
			panel_path = plotoutput / f"{name}_validate_panel_OLS_Ridge.png"
			fig.savefig(panel_path)
			plt.close(fig)

	# writes the summary
	outputdf = pd.DataFrame(results) #final dataframe from results list
	outputdf.to_csv(outputdir / "regression_summary.csv", index=False) #writes the dataframe to a csv file
	print("Wrote regression summary to", outputdir / "regression_summary.csv") #prints out where the csv was written
	print("Wrote plots to", outputdir / "plots") #print out where the plots were written

#Further plot to put both ridge and ols atop each other to look for differences (they are basically unnoticable)
def plot_compare(per_station, outputdir, station_name=None): 
	plotoutput = outputdir / "plots" #we already have the plot output dir from earlier so this can go here too 
	datesplit = pd.Timestamp("2000-01-01") #again sets a date split

	for name, df in per_station.items(): #for loop so we run the same process with all 3 stations
		df = df.dropna(subset=["target", "MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]).copy() #drop rows with nans in our used columns
		pre = df[df.index < datesplit] #splits the dataframe into our given time setups
		post = df[df.index >= datesplit]

		train_df, test_df = pre, post #sets the train and test frames to the respective timeframes too
		X_train = train_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #x value training set
		y_train = train_df["target"].values #the y value is the "target"
		X_test = test_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values #then the test sets
		y_test = test_df["target"].values #these test value are what we compare our predictions to

		# OLS
		ols = LinearRegression() #we can just redo our linear regression here for ols
		ols.fit(X_train, y_train) #and fit it with the training data
		ypred_ols = ols.predict(X_test) #then just predict on the test data
		#Then we can do ridge here
		ridge_pipe = make_pipeline(StandardScaler(), Ridge()) #We use the pipeline to scale the data first then do ridge regression
		ridge_pipe.fit(X_train, y_train) #fit the ridge model with training data
		if hasattr(ridge_pipe, "named_steps") and "ridge" in ridge_pipe.named_steps: #check for named steps and ridge step
			ridge_est = ridge_pipe.named_steps["ridge"] #set ridge estimator to ridge step
		else: #or 
			ridge_est = ridge_pipe #set ridge estimator to the pipeline itself
		ypred_ridge = ridge_pipe.predict(X_test) #so we can predict on the test data with the ridge model

		##################SCATTER SECTION######################
		#Safeguard for r2 values
		try: #These try's essentially calculate the r2 values and blanks them if something goes wrong
			r2_ols = float(r2_score(y_test, ypred_ols)) if len(y_test) > 0 else float("nan")
		except Exception:
			r2_ols = float("nan")
		try:
			r2_ridge = float(r2_score(y_test, ypred_ridge)) if len(y_test) > 0 else float("nan")
		except Exception:
			r2_ridge = float("nan")

		# Panel creation, when we use the station argument
		if station_name and name == station_name: #if the station name argument is used and matches the current station name
			display_name = STATION_ACTUALS.get(name, name.replace('WS', '').replace('_', ' ')) #uses the mapping of actual names here too
			mn = min(y_test.min(), ypred_ols.min(), ypred_ridge.min()) #min and max values for axis limits
			mx = max(y_test.max(), ypred_ols.max(), ypred_ridge.max())
			fig, axes = plt.subplots(1, 2, figsize=(12, 8)) #sets up a subplot in a 1x2 format (and gives our figure size)
			ax0 = axes[0] #sets axes for plots within the subplot
			ax1 = axes[1]
			#Basic scatter plot setup, this one for OLS same as previous ones in printing R2 values and all just properly printing them
			ax0.scatter(y_test, ypred_ols, s=20, alpha=0.7, color="C0")
			ax0.plot([mn, mx], [mn, mx], color="k", linestyle="--")
			ax0.set_xlabel("Observed 2m Surface Temperature, C")
			ax0.set_ylabel("Predicted 2m Surface Temperature, C")
			ax0.set_title(f"OLS: Observed vs Predicted — {display_name}")
			ax0.text(0.02, 0.98, f"R2_test: {r2_ols:.3f}" if not np.isnan(r2_ols) else "R2_test: nan",
				transform=ax0.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
			#Repeat with Ridge model
			ax1.scatter(y_test, ypred_ridge, s=20, alpha=0.7, color="C1")
			ax1.plot([mn, mx], [mn, mx], color="k", linestyle="--")
			ax1.set_xlabel("Observed 2m Surface Temperature, C")
			ax1.set_ylabel("Predicted 2m Surface Temperature, C")
			ax1.set_title(f"Ridge: Observed vs Predicted — {display_name}")
			ax1.text(0.02, 0.98, f"R2_test: {r2_ridge:.3f}" if not np.isnan(r2_ridge) else "R2_test: nan",
				transform=ax1.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
			#final layout and a quick save
			fig.suptitle(f"{display_name} — OLS vs Ridge (Test)")
			fig.tight_layout(rect=[0, 0.03, 1, 0.97])
			panel_path = plotoutput / f"{name}_panel_OLS_Ridge.png"
			fig.savefig(panel_path)
			plt.close(fig)
			continue

		#This scatter setup when there's no station arg
		fig, ax = plt.subplots(figsize=(6, 6)) #I used subplot here but it really doesnt need it it's just a basic single plot
		ax.scatter(y_test, ypred_ols, s=20, alpha=0.7, label="OLS predicted", color="C0") #scatter for ols
		ax.scatter(y_test, ypred_ridge, s=20, alpha=0.7, label="Ridge predicted", color="C1") #scatter for ridge
		mn = min(y_test.min(), ypred_ols.min(), ypred_ridge.min()) #min and max values for axis limits
		mx = max(y_test.max(), ypred_ols.max(), ypred_ridge.max())
		ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="1:1") #our perfect line
		ax.set_xlabel("Observed 2m Surface Temperature, C")
		ax.set_ylabel("Predicted 2m Surface Temperature, C")
		display_name = STATION_ACTUALS.get(name, name.replace('WS', '').replace('_', ' '))
		ax.set_title(f"{display_name} compare OLS vs Ridge: Observed vs Predicted")
		ax.text(0.02, 0.98, f"OLS R2: {r2_ols:.3f}\nRidge R2: {r2_ridge:.3f}", transform=ax.transAxes, #R2 values
			fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
		ax.legend()
		fig.tight_layout()
		scatter_path = plotoutput / f"{name}_compare_OLS_vs_Ridge_obs_vs_pred.png"
		fig.savefig(scatter_path)
		plt.close(fig)

		#Timeseries plots, idk if we need these still they are cool but kind of unnecessary but also kind of help with visualisation
		fig, ax = plt.subplots(figsize=(10, 4)) #same paramter setup as other plots
		ax.plot(test_df.index, y_test, label="observed", color="k")
		ax.plot(test_df.index, ypred_ols, label="OLS predicted", color="C0", linestyle="-")
		ax.plot(test_df.index, ypred_ridge, label="Ridge predicted", color="C1", linestyle="--")
		ax.set_ylabel("2m Surface Temperature, C")
		ax.set_title(f"{name} compare OLS vs Ridge Test Time Series")
		ax.text(0.02, 0.98, f"OLS R2: {r2_ols:.3f}\nRidge R2: {r2_ridge:.3f}", transform=ax.transAxes,
			fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
		ax.legend()
		fig.tight_layout()
		ts_path = plotoutput / f"{name}_compare_OLS_vs_Ridge_timeseries.png"
		fig.savefig(ts_path)
		plt.close(fig)

	print("Wrote comparison plots to", plotoutput) #confirmation

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
	parser.add_argument("--compare", action="store_true", help="Will put the stations atop each other") #This was to put the two types atop of each other
	parser.add_argument("--station", "-s", default=None, help="Just to focus on 1 station at once in compare") #station arg was breaking think I fixed
	args = parser.parse_args() #parses the arguments

	base = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent #sets the base directory to either the arg provided or the current script directory

	if not args.run: #if nothings used drop this messge, we could just default to true
		print("use the shown args (--run) to properly start") 
		return 

	data = load_data(base) #loads the data from the base directory
	pstation = reg_tables(data["mei"], data["aaoi"], data["stations"]) #prepares the regression tables
	outputdirec = base / "regression_out" #output directory for regression results

	if args.compare:
		plot_compare(pstation, outputdirec, station_name=args.station) #if compares gone with run just this
		return

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

