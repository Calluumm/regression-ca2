#import modules you'll have to pip install missing ones
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#set to file names that you guys use i simplified mine to these
AAOI_FNAME = "aaoi.csv"
MEI_FNAME = "mei.csv"
ANTO_FNAME = "AntoWS.csv"
PUERTO_FNAME = "PuertoWS.csv"
QUIN_FNAME = "QuinWS.csv"

#data loads
def load_data(base):
	meipath = base / MEI_FNAME
	meivalues = []
	with meipath.open("r", encoding="utf-8") as f:
		for line in f:
			s = line.strip()
			if not s or s.startswith("#"):
				continue
			meivalues.append(float(s))
	mei_idx = pd.date_range(start="1979-01-01", periods=len(meivalues), freq="MS")
	mei = pd.Series(data=meivalues, index=mei_idx, name="MEI")

	aaoipath = base / AAOI_FNAME
	aaoi = pd.read_csv(aaoipath, header=None, names=["date", "value"], parse_dates=[0], index_col=0)
	aaoi["value"] = aaoi["value"].astype(str).str.replace("D", "E").astype(float)
	aaoi = aaoi["value"].rename("AAOI")

	stations = {}
	for fname in [ANTO_FNAME, PUERTO_FNAME, QUIN_FNAME]:
		p = base / fname
		df = pd.read_csv(p, parse_dates=[0])
		if "time" in df.columns:
			df = df.set_index("time")
		df["t2m_C"] = df["t2m"] - 273.15
		df["wind_ms"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
		stations[fname.replace('.csv', '')] = df[["t2m_C", "wind_ms"]].copy()

	return {"mei": mei, "aaoi": aaoi, "stations": stations}

#regression table making
def reg_tables(mei, aaoi, stations):
	base_df = pd.concat([mei, aaoi], axis=1)

	combined = []
	for name, df in stations.items():
		df2 = df.rename(columns={"t2m_C": f"t2m_{name}", "wind_ms": f"wind_{name}"})
		combined.append(df2)
	combined_df = pd.concat(combined, axis=1)
	t2m_cols = [c for c in combined_df.columns if c.startswith("t2m_")]
	wind_cols = [c for c in combined_df.columns if c.startswith("wind_")]
	combined_df["regional_t2m_mean"] = combined_df[t2m_cols].mean(axis=1)
	combined_df["regional_wind_mean"] = combined_df[wind_cols].mean(axis=1)

	merged = base_df.join(combined_df[["regional_t2m_mean", "regional_wind_mean"]], how="inner")

	per_station = {}
	for name in stations.keys():
		tcol = f"t2m_{name}"
		wcol = f"wind_{name}"
		df = merged[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].join(combined_df[[tcol, wcol]], how="left")
		df = df.rename(columns={tcol: "target", wcol: "wind_station"})
		per_station[name] = df
	return per_station


def reporting(per_station, outputdir, use_ridge=False):
	results = []
	datesplit = pd.Timestamp("2000-01-01")
	outputdir.mkdir(parents=True, exist_ok=True)
	plotoutput = outputdir / "plots"
	plotoutput.mkdir(parents=True, exist_ok=True)

	for name, df in per_station.items():
		df = df.dropna(subset=["target", "MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]).copy()
		pre = df[df.index < datesplit]
		post = df[df.index >= datesplit]
		for period_name, period_df in (("pre2000", pre), ("post2000", post)):
			if period_df.shape[0] < 10:
				continue
			X = period_df[["MEI", "AAOI", "regional_t2m_mean", "regional_wind_mean"]].values
			y = period_df["target"].values

			for model_name, ModelCls in (("OLS", LinearRegression), ("Ridge", Ridge)):
				model = ModelCls()
				model.fit(X, y)
				ypred = model.predict(X)
				r2 = float(r2_score(y, ypred))
				coefs = list(model.coef_)
				intercept = float(model.intercept_)
				results.append({
					"station": name,
					"period": period_name,
					"model": model_name,
					"n_rows": period_df.shape[0],
					"intercept": intercept,
					"coef_MEI": coefs[0],
					"coef_AAOI": coefs[1],
					"coef_reg_t2m": coefs[2],
					"coef_reg_wind": coefs[3],
					"R2": r2,
				})

				rpt = outputdir / f"report_{name}_{period_name}_{model_name}.txt"
				with rpt.open("w", encoding="utf-8") as f:
					f.write(f"Station: {name}\nPeriod: {period_name}\nModel: {model_name}\nRows: {period_df.shape[0]}\n\n")
					f.write(f"Intercept: {intercept}\n")
					f.write(f"coef_MEI: {coefs[0]}\n")
					f.write(f"coef_AAOI: {coefs[1]}\n")
					f.write(f"coef_reg_t2m: {coefs[2]}\n")
					f.write(f"coef_reg_wind: {coefs[3]}\n")
					f.write(f"R2: {r2}\n")

				fig, ax = plt.subplots(figsize=(6, 6))
				ax.scatter(y, ypred, s=20, alpha=0.7)
				mn = min(y.min(), ypred.min())
				mx = max(y.max(), ypred.max())
				ax.plot([mn, mx], [mn, mx], color="k", linestyle="--", label="1:1")
				ax.set_xlabel("Observed t2m_C")
				ax.set_ylabel("Predicted t2m_C")
				ax.set_title(f"{name} {period_name} {model_name} Observed vs Predicted")
				ax.legend()
				fig.tight_layout()
				scatter_path = plotoutput / f"{name}_{period_name}_{model_name}_obs_vs_pred.png"
				fig.savefig(scatter_path)
				plt.close(fig)

				fig, ax = plt.subplots(figsize=(10, 4))
				ax.plot(period_df.index, y, label="observed")
				ax.plot(period_df.index, ypred, label="predicted")
				ax.set_ylabel("t2m_C")
				ax.set_title(f"{name} {period_name} {model_name} Time Series")
				ax.legend()
				fig.tight_layout()
				ts_path = plotoutput / f"{name}_{period_name}_{model_name}_timeseries.png"
				fig.savefig(ts_path)
				plt.close(fig)

	outputdf = pd.DataFrame(results)
	outputdf.to_csv(outputdir / "regression_summary.csv", index=False)
	print("Wrote regression summary to", outputdir / "regression_summary.csv")
	print("Wrote plots to", outputdir / "plots")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--rows", type=int, default=10)
	parser.add_argument("--run", action="store_true")
	parser.add_argument("--dir", "-d", default=None)
	args = parser.parse_args()

	base = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent

	if not args.run:
		print("run with --run for regression stuff")
		return

	data = load_data(base)
	pstation  = reg_tables(data["mei"], data["aaoi"], data["stations"])
	outputdirec = base / "regression_out"
	reporting(pstation , outputdirec)


if __name__ == "__main__":
	main()