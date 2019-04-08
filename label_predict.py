import preprocess
from metadata import *
import gov_data
import numpy as np
import json
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def get_gov_curated(all_resources):
	tids = []
	for resource in all_resources:
		for each_data in resource.data_files:
				tids.append(resource.rs_id + ':' + each_data.df_id)
	f = open(gov_curated_path, 'r')
	X = []  # curated
	y = []
	count = 0
	total = len(tids)
	for line in f:
		tid = list(json.loads(line).keys())[0]
		if tid not in tids:
			continue
		count += 1
		print('find {0} of {1}'.format(count, total))
		if tid == 'feature_ids':
			break
		table_cols = list(json.loads(line).values())[0]
		headers = [each[0].lower() for each in table_cols]
		if len(headers) < 2:
			continue
		for col, features in table_cols:
			X.append(features)
			y.append(col.lower())
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	X = imp.fit_transform(X)
	maxabs_scale = preprocessing.MaxAbsScaler()
	X = maxabs_scale.fit_transform(X)
	return X,y

def main():
	# curated
	X,y = get_gov_curated(rand_gov_resources)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	clf = preprocess.train_RF_model(preprocess.normalize_X(X_train), y_train, n_estimators=25, n_jobs=4,freq_filter=1)
	preprocess.eval_model(preprocess.normalize_X(X_test), np.array(y_test), clf, freq_filter=1)
	preprocess.eval_model(preprocess.normalize_X(X_test), np.array(y_test), clf, freq_filter=1, only_seen=True)


if __name__ == "__main__":
	main()