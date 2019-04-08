import gov_data
from metadata import *
import pandas as pd 
import multiprocessing  as mp
from nltk.tokenize import ToktokTokenizer
import json
import numpy
from collections import Counter
from scipy import signal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from gensim.models.fasttext import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
from nltk.corpus import brown


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class Dictionary(object):
	'''
	TODO:
	a lot of cases to handle the errors
	1. no such file file (due to download errors)
	2. have the file with data.csv, but is not csv file -> col errors
	'''
	def __init__(self):
		self.wl = set()
		for each in brown.words():
			each = each.lower()
			if each.isalpha() and (each not in self.wl):
					self.wl.add(each)
		self.toktok = ToktokTokenizer()

	def isDW(self,label):
		tokens = self.toktok.tokenize(label)
		flag = True
		for token in tokens:
			if token.lower() not in self.wl:
				flag = False
				break
		return flag


def extract_gov_fdict(all_resources,fdict_path = gov_data_fdict_path,tid_type = 'cat_id',restrict_resource = False):
    #extracting features:
	#table_id;label,curated_features,content;label,curated_features...
	f = open(fdict_path,'w')
	#all_resources = gov_data.read_resources()
	all_resources = gov_data.wrong_csv(all_resources)
	all_resources = list(filter(lambda x:x.status,all_resources))
	if restrict_resource:
		all_resources = gov_data.select_resources(all_resources,fsize = 50,rs_ct = len(all_resources))
	pool = mp.Pool()
	total = len(all_resources)
	count = 0
	toktok = ToktokTokenizer()
	for resource in all_resources:
		print("processing {0}-th resource".format(count))
		for each_data in resource.data_files:
			try:
				if tid_type == 'cat_id':
					tid = resource.rs_id + ':' + each_data.df_id
				elif tid_type == 'path':
					tid = resource.path + '/' + each_data.df_id
				d_path = each_data.path + '/data.csv'
				df = pd.read_csv(d_path, delimiter=',', quotechar='"',dtype=str,na_filter = True )
				cols = df.columns
				contents = [df[each_col].dropna().tolist() for each_col in cols]
				print("extract content finished")
				cols_features = pool.map(gov_data.get_col_features, contents)
				all_col_features = list(zip(cols,cols_features))
				for i in range(len(all_col_features)):
					if all_col_features[i][1]:
						all_col_features[i][1]['content'] =  toktok.tokenize(' '.join(contents[i]))
				all_col_features = list(filter(lambda x:x[1],all_col_features))
				f.write(json.dumps({tid:all_col_features},cls=MyEncoder) + '\n')
			except Exception as e:
				print(e)
		count += 1
		print("finish {0} out of {1}".format(count, total))
	f.close()
	return all_resources




def check_format(t):
	'''
	make constraints on tables
	'''
	min_cols = 4
	min_rows = 6
	if t['numCols'] < min_cols or t['numDataRows'] < min_rows:
		return False
	return True

def extract_wiki_fdict():
	f_count = 0
	# for each wiki table, get header name, and corresponding content
	f = open(wiki_path,'r')
	f_dest = open(wiki_fdict_path,'w')
	toktok = ToktokTokenizer()
	tid = 0
	pool = mp.Pool()
	for line in f:
		tid += 1
		t = json.loads(line)
		if not check_format(t):
			continue
		try:
			# header process
			header_iter = iter(t['tableHeaders'][-1])
			header_span = []
			header_content = dict()
			header_bows = dict()
			header_idx = 0
			for each_header in header_iter:
				html_desc =each_header['tdHtmlString']
				span = int(html_desc.split('colspan="')[1].split('"')[0])
				header_span.append((each_header['text'],span))
				header_content[header_idx] = []
				header_bows[header_idx] = []
				header_idx += 1
				if span != 1:
					for skip_num in range(span-1):
						next(header_iter) 
			# content process
			for row in t['tableData']:
				global_col_index = 0
				header_idx = 0
				for header,span in header_span:
					for idx in range(span):
						if row[global_col_index]['text'] != '':
							header_content[header_idx].append(row[global_col_index]['text'])
							header_bows[header_idx].extend(toktok.tokenize(row[global_col_index]['text']))
						global_col_index += 1
					header_idx += 1
		except:
			continue
		#combine header and features
		cols_features = pool.map(gov_data.get_col_features, list(header_content.values()))
		all_col_features = list(zip([each[0] for each in header_span],cols_features))
		for i in range(len(all_col_features)):
			if all_col_features[i][1]:
				all_col_features[i][1]['content'] = header_bows[i]	
		all_col_features = list(filter(lambda x:x[1],all_col_features))
		f_dest.write(json.dumps({tid:all_col_features},cls=MyEncoder) + '\n')
		print("finishing {0}".format(f_count))
		f_count += 1


def extract_wiki_curated_features(hist_len = 20):
	# process histogram and unigram features, produce vector features
	'''
	special handle for histogram and unigram
	'''
	#first, building unigram vocabulary
	f_dict = open(wiki_fdict_path,'r')
	print("find unigrams")
	#start_time = time.time()
	unigrams = Counter()
	for line in f_dict:
		table_dict = json.loads(line)
		for tid in table_dict:
			for each in table_dict[tid]:
				unigrams.update(each[1]['unigram'])
	f_dict.close()
	unigrams = sorted(unigrams.keys())
	#concatenate features
	rs_ct = 0
	col_f_ids = []
	hist_flag = False
	uni_flag = False
	f_ids = sorted(list(table_dict[tid][0][1].keys()))
	f_ids.remove('content')
	f_final = open(wiki_curated_path,'w')
	f_dict = open(wiki_fdict_path,'r')
	for line in f_dict:
		table_dict = json.loads(line)
		all_col_features = []
		for tid in table_dict:
			for col,feature_dict in table_dict[tid]:
				f_vec = []
				for f_id in f_ids:
					#concatenate unigram features
					if f_id == 'unigram':
						if not uni_flag:
							col_f_ids.extend('unigrams')
							uni_flag = True
						unigram = feature_dict[f_id]
						uni_features = list(map(lambda t:0 if t not in unigram else unigram[t],unigrams))
						f_vec.extend(uni_features)
					#contatenate histogram features
					elif f_id == 'hist':
						if not hist_flag:
							col_f_ids.extend( ['h'+str(i) for i in range(hist_len)])
							hist_flag = True
						col_hist = feature_dict[f_id]
						f_vec.extend(signal.resample(sorted(col_hist.values()), hist_len))
					else:
						if f_id not in col_f_ids:
							col_f_ids.append(f_id)
						f_vec.append(feature_dict[f_id])
					#next_f_time = time.time() - last_f_time
					#print("time for feature {0} : {1}".format(f_id,next_f_time))
					#last_f_time = time.time()
				rs_ct +=1 
				if rs_ct%100 == 0:
					print("final transformation finished {0} out of {1}: {2}".format(rs_ct,total_rs,rs_ct/total_rs))
				if len(col_f_ids) == 55191:
					break
				all_col_features.append((col,f_vec))
		f_final.write(json.dumps({tid:all_col_features}) + '\n')
	f_final.write(json.dumps({'feature_ids':col_f_ids}) + '\n')


def extract_gov_curated_features(fdict_path,curated_path,hist_len = 20):
	# process histogram and unigram features, produce vector features
	'''
	special handle for histogram and unigram
	'''
	#first, building unigram vocabulary
	f_dict = open(fdict_path,'r')
	print("find unigrams")
	#start_time = time.time()
	unigrams = Counter()
	total_rs = 0
	for line in f_dict:
		total_rs += 1
		table_dict = json.loads(line)
		for tid in table_dict:
			for each in table_dict[tid]:
				unigrams.update(each[1]['unigram'])
	f_dict.close()
	unigrams = sorted(unigrams.keys())
	#concatenate features
	rs_ct = 0
	col_f_ids = []
	hist_flag = False
	uni_flag = False
	f_ids = sorted(list(table_dict[tid][0][1].keys()))
	f_ids.remove('content')
	f_final = open(curated_path,'w')
	f_dict = open(fdict_path,'r')
	for line in f_dict:
		table_dict = json.loads(line)
		all_col_features = []
		for tid in table_dict:
			for col,feature_dict in table_dict[tid]:
				f_vec = []
				for f_id in f_ids:
					#concatenate unigram features
					if f_id == 'unigram':
						if not uni_flag:
							col_f_ids.extend(unigrams)
							uni_flag = True
						unigram = feature_dict[f_id]
						uni_features = list(map(lambda t:0 if t not in unigram else unigram[t],unigrams))
						f_vec.extend(uni_features)
					#contatenate histogram features
					elif f_id == 'hist':
						if not hist_flag:
							col_f_ids.extend( ['h'+str(i) for i in range(hist_len)])
							hist_flag = True
						col_hist = feature_dict[f_id]
						f_vec.extend(signal.resample(sorted(col_hist.values()), hist_len))
					else:
						if f_id not in col_f_ids:
							col_f_ids.append(f_id)
						f_vec.append(feature_dict[f_id])
					#next_f_time = time.time() - last_f_time
					#print("time for feature {0} : {1}".format(f_id,next_f_time))
					#last_f_time = time.time()
				rs_ct +=1 
				if rs_ct%100 == 0:
					print("final transformation finished {0} out of {1}: {2}".format(rs_ct,total_rs,rs_ct/total_rs))
				if len(col_f_ids) == 55191:
					break
				all_col_features.append((col,f_vec))
		f_final.write(json.dumps({tid:all_col_features}) + '\n')
	f_final.write(json.dumps({'feature_ids':col_f_ids}) + '\n')



def topn_accuracy(clf,X_test,y_test,topn = 10):
	preds = []
	iters = int(X_test.shape[0]/1000)
	for i in range(iters+1):
		preds.append(clf.predict_proba(X_test[i*1000:(i*1000+1000)]))
	probs = np.concatenate(preds)
	correct_ct = 0
	for i in range(len(probs)):
		prob = probs[i]
		topn_rs =  np.argpartition(prob, -topn)[-topn:]
		if y_test[i] in clf.classes_[topn_rs]:
			correct_ct += 1
	return correct_ct*1.0/X_test.shape[0]



def eval_model(X_test,y_test,clf,freq_filter = 0,only_seen = False,high_prob = 0,topn=1):
	#filter testing set
	if freq_filter != 0:
		label_ct = Counter(y_test)
		new_X = []
		new_y = []
		for idx in range(len(y_test)):
			if label_ct[y_test[idx]] <= freq_filter:
				continue
			new_X.append(X_test[idx])
			new_y.append(y_test[idx])
		X_test = np.array(new_X)
		y_test = new_y
	if only_seen:
		new_X = []
		new_y = []
		for idx in range(len(y_test)):
			if y_test[idx] in clf.classes_:
				new_X.append(X_test[idx])
				new_y.append(y_test[idx])
		X_test = np.array(new_X)
		y_test = new_y
	# begin evaluation
	preds = []
	iters = int(X_test.shape[0]/1000)
	for i in range(iters+1):
		preds.append(clf.predict(X_test[i*1000:(i*1000+1000)]))
	preds = np.concatenate(preds)
	#filter low prob ?
	if high_prob != 0:
		preds_prob = []
		iters = int(X_test.shape[0]/1000)
		for i in range(iters+1):
			preds_prob.append(clf.predict_proba(X_test[i*1000:(i*1000+1000)]))
		preds_prob = np.concatenate(preds_prob)
		preds_prob = [max(each) for each in preds_prob]
		new_preds = []
		new_y = []
		for idx in range(len(y_test)):
			if preds_prob[idx] > high_prob:
				new_preds.append(preds[idx])
				new_y.append(y_test[idx])
		preds = np.array(new_preds)
		y_test = new_y
	macro = precision_recall_fscore_support(y_test, preds, average='macro')
	micro = precision_recall_fscore_support(y_test, preds, average='micro')
	#topn_acc = [ topn_accuracy(clf,X_test,y_test,topn = i + 1) for i in range(topn)]
	print('macro: {0} \n micro:{1} \n '.format(macro,micro))
	return macro,micro

def normalize_X(X):
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	X = imp.fit_transform(X)
	maxabs_scale = preprocessing.MaxAbsScaler()
	X = maxabs_scale.fit_transform(X)
	return X

def train_RF_model(X,y,n_estimators=25,n_jobs=4,freq_filter=0):
	if freq_filter != 0:
		label_ct = Counter(y)
		new_X = []
		new_y = []
		for idx in range(len(y)):
			if label_ct[y[idx]] <= freq_filter:
				continue
			new_X.append(X[idx])
			new_y.append(y[idx])
		X = new_X
		y = new_y
	clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=n_jobs)
	clf.fit(X,y)
	return clf


def get_all_gov_data(rand_gov_resources):
	tids = []
	for resource in rand_gov_resources:
		for each_data in resource.data_files:
				tids.append(resource.rs_id + ':' + each_data.df_id)
	f_curated = open(gov_curated_path,'r')
	f_bows = open(gov_data_fdict_path,'r')
	curated = []
	contents = []
	y = []
	count = 0
	total = len(tids)
	for line in f_curated:
		line2 = f_bows.readline()
		tid = list(json.loads(line).keys())[0]
		if tid not in tids:
			continue
		count += 1
		print('find {0} of {1}'.format(count,total))
		tids.remove(tid)
		if len(tids) == 0:
			break
		table_cols = json.loads(line)[tid]
		col_dicts = json.loads(line2)[tid]
		headers = [each[0].lower() for each in table_cols]
		if len(headers) < 2:
			continue
		if len(table_cols) != len(col_dicts):
			print("cannot align")
			continue
		for i in range(len(table_cols)):
			col = table_cols[i][0]
			features = table_cols[i][1]
			curated.append(features)
			contents.append(col_dicts[i][1]['content'])
			#features = np.hstack((features,context_embed))
			y.append(col.lower())
	return curated,contents,y


def get_all_wiki_data(total = 5000):
	f_curated = open(wiki_curated_path,'r')
	f_bows = open(wiki_fdict_path,'r')
	curated = []
	contents = []
	y = []
	count = 0
	for line in f_curated:
		line2 = f_bows.readline()
		tid = list(json.loads(line).keys())[0]
		if count > total:
			break
		count += 1
		print('find {0} '.format(count))
		table_cols = json.loads(line)[tid]
		col_dicts = json.loads(line2)[tid]
		headers = [each[0].lower() for each in table_cols]
		if len(headers) < 2:
			continue
		if len(table_cols) != len(col_dicts):
			print("cannot align")
			continue
		for i in range(len(table_cols)):
			col = table_cols[i][0]
			features = table_cols[i][1]
			curated.append(features)
			contents.append(col_dicts[i][1]['content'])
			#features = np.hstack((features,context_embed))
			y.append(col.lower())
	return curated,contents,y


def compare_models(contents,curated,y,filter_col = 0):
	label_ct = Counter(y)
	new_contents = []
	new_curated = []
	new_y = []
	for idx in range(len(y)):
		if label_ct[y[idx]] <= filter_col:
			continue
		if len(contents[idx]) < 1:
			continue
		new_contents.append(contents[idx])
		new_curated.append(curated[idx])
		new_y.append(y[idx])
	# construct bow features
	bows = []
	vectorizer = CountVectorizer()
	for idx,col in enumerate(new_contents):
			tokens =  [str(each) for each in new_contents[idx]]
			bows.append(' '.join(new_contents[idx]))
	bows = vectorizer.fit_transform(bows)
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(bows)
	SVD = decomposition.TruncatedSVD(n_components=300)
	SVD.fit(tfidf)
	tfidf = SVD.transform(tfidf)
	#combine features
	combines = []
	for i in range(len(new_curated)):
		combines.append(np.hstack((new_curated[i],tfidf[i])))
	#preprocesing
	combines = np.array(combines)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	combines = imp.fit_transform(combines)
	maxabs_scale = preprocessing.MaxAbsScaler()
	combines = maxabs_scale.fit_transform(combines)
	# split data
	X_train, X_test, y_train, y_test = train_test_split(combines, new_y, train_size = 0.7, random_state=42)
	## eval combine
	print("combined results:")
	clf = train_RF_model(X_train,y_train,n_estimators=25,n_jobs=4)
	macro,micro,topn_acc = eval_model(X_test,y_test,clf)
	combine_rs = dict()
	combine_rs['topn accuracy'] = topn_acc
	combine_rs['macro'] = macro
	combine_rs['micro'] = micro
	print("curated results:")
	## eval curated:
	clf = train_RF_model(X_train[:,0:len(curated[0])],y_train,n_estimators=25,n_jobs=4)
	macro,micro,topn_acc = eval_model(X_test[:,0:len(curated[0])],y_test,clf)
	curated_rs = dict()
	curated_rs['topn accuracy'] = topn_acc
	curated_rs['macro'] = macro
	curated_rs['micro'] = micro
	print("BoWs results:")
	##eval bows
	clf = train_RF_model(X_train[:,len(curated[0]):],y_train,n_estimators=25,n_jobs=4)
	macro,micro,topn_acc = eval_model(X_test[:,len(curated[0]):],y_test,clf)
	bows_rs = dict()
	bows_rs['topn accuracy'] = topn_acc
	bows_rs['macro'] = macro
	bows_rs['micro'] = micro
	return combine_rs,curated_rs,bows_rs

if __name__ == '__main__':
	pass

