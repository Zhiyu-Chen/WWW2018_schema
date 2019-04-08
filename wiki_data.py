from __future__ import division
from metadata import *
import json
from gov_data import get_col_features
import pickle
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import nltk
import requests
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import random

import csv
from util import tagged_columns
from util import Dictionary

def corenlp_tokenizer(data,normalizing = True):
    '''
    parse a sentence of a review, (normalzied)
    Input: a sentence 
    Output: nn,adj,dependecy pairs,
    '''
    url ='http://localhost:9000/?properties={%22annotators%22%3A%22tokenize%2Cssplit%22%2C%22outputFormat%22%3A%22json%22}'
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, data=data.encode('utf-8'), headers=headers)
    tokens = []
    for each in r.json(strict=False)['sentences'][0]['tokens']:
            if normalizing:
                tokens.append(porter.stem(wnl.lemmatize(each['word'].lower())))
            else:
                tokens.append(each['word'])
    return tokens



def check_format(t):
	'''
	make constraints on tables
	'''
	min_cols = 4
	min_rows = 6
	if t['numCols'] < min_cols or t['numDataRows'] < min_rows:
		return False
	return True

def extract_wiki_features(wiki_feature_path,wiki_bow_path):
	f_count = 0
	# for each wiki table, get header name, and corresponding content
	f = open(wiki_path,'r')
	f_dest = open(wiki_feature_path,'w')
	f_bow = open(wiki_bow_path,'w')
	toktok = ToktokTokenizer()
	for line in f:
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
		for col, f_dict,bows in zip([each[0] for each in header_span],map(get_col_features,header_content.values()),header_bows.values()):
			if f_dict:
				f_dict['_id'] = t['_id']
				f_dest.write(json.dumps({col:f_dict}) + '\n')
				f_bow.write(json.dumps({col:bows}) + '\n')
		print("finishing {0}".format(f_count))
		f_count += 1


def split_wiki_final_features():
	count = 0
	f = open(wiki_final_features,'r')
	patch_num = 0
	for line in f:
		if count % 100000 == 0:
			f_patch = open(wiki_final_features_patches + '_' + str(patch_num) + '.json','w')
			patch_num += 1
		f_patch.write(line)
		count +=1


def read_wiki_final_features(wiki_final_features):
	f = open(wiki_final_features,'r')
	all_features = []
	labels = []
	cols = []
	rows = []
	row_idx = 0
	feature_lenth = 55191
	for line in f:
		features = json.loads(line).items()[0][1]
		labels.append(json.loads(line).items()[0][0])
		for col_idx,f_value in filter(lambda x:x[1] != 0,[(each[0],each[1]) for each in zip(range(feature_lenth),features) ]):
			cols.append(col_idx)
			rows.append(row_idx)
			all_features.append(f_value)
		if row_idx % 1000 == 0:
			print(row_idx)
		row_idx += 1
	#col_f_matrix = csr_matrix((all_features, (rows, cols)), shape=(row_idx, 55191))
	return all_features,labels 


def preprocess_col_features(col_f_matrix,dim=300):
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	X = imp.fit_transform(col_f_matrix)
	del col_f_matrix
	maxabs_scale = preprocessing.MaxAbsScaler()
	X = maxabs_scale.fit_transform(X)
	SVD = decomposition.TruncatedSVD(n_components=dim)
	SVD.fit(X)
	X = SVD.transform(X)
	return X


def col_label_accumulative(labels,topn = 500):
	# plot accumulative count of columns. X:top N Y: accumulative percentage
	col_counter = Counter(labels)
	col_counter.pop('')
	total_count = sum(col_counter.values())
	acc_percent = []
	acc_count = 0
	for idx,pair in enumerate(col_counter.most_common()):
		acc_count += pair[1]
		acc_percent.append(acc_count/total_count)
		print(idx,pair[0],pair[1])
		if idx > topn:
			break
	plt.clf()
	plt.figure(figsize=(6,3))
	dont_return = plt.plot(range(len(acc_percent)),acc_percent)
	plt.xlabel('top n labels')
	plt.ylabel('accumulative percentage')
	plt.savefig('col_acc')


def report_result(y_test,preds):
	macro = f1_score(y_test,preds,average='macro')
	micro = f1_score(y_test,preds,average='micro')
	accuracy = accuracy_score(y_test, preds)
	print("macro f1 : {0}".format(macro))
	print("micro f1 : {0}".format(micro))
	print("accuracy : {0}".format(accuracy))



def topn_accuracy(clf,X_test,y_test,topn = 10):
	iters = int(X_test.shape[0]/10000)
	correct_ct = 0
	sample_idx = 0
	for i in range(iters+1):
		probs = clf.predict_proba(X_test[i*10000:(i*10000+10000)])
		for j in range(len(probs)):
			prob = probs[j]
			topn_rs =  np.argpartition(prob, -topn)[-topn:]
			if y_test[sample_idx] in clf.classes_[topn_rs]:
				correct_ct += 1
			sample_idx += 1
	return correct_ct*1.0/len(X_test)


def generate_Human_report(clf,X_test,y_test,index_dict,trans,fname ,sample_n = 500,theta=0.5):
	idx2id = dict(zip(index_dict.values(),index_dict.keys()))
	indices = range(X_test.shape[0])
	random.shuffle(indices)
	target_x = X_test[indices[:sample_n]]
	gold_y = y_test[indices[:sample_n]]
	target_y = clf.predict(target_x)
	probs = clf.predict_proba(target_x)
	gold_y =  map(lambda x:idx2id[x],gold_y)
	target_y =  map(lambda x:idx2id[x],target_y) 
	gold_label = trans.inverse_transform(gold_y)
	target_label = trans.inverse_transform(target_y)
	csvWriter = csv.writer(open(fname, 'w'))
	csvWriter.writerow(['Ground truth','prediction','probability'])
	for i in range(len(gold_label)):
		if gold_label[i] == target_label[i] or max(probs[i]) < theta:
			continue
		csvWriter.writerow([gold_label[i].encode('utf-8') , target_label[i].encode('utf-8') , max(probs[i])])



def generate_csv_results(clf,X_test,y_test,fname,sample_n = 2000):
	preds = []
	iters = int(X_test.shape[0]/10000)
	for i in range(iters+1):
		preds.append(clf.predict(X_test[i*10000:(i*10000+10000)]))
	rs = np.concatenate(preds)
	probs = []
	iters = int(X_test.shape[0]/10000)
	for i in range(iters+1):
		probs.append(clf.predict_proba(X_test[i*10000:(i*10000+10000)]))
	probs = np.concatenate(probs)
	probs = np.array([max(each) for each in probs])
	ranked_idx = probs.argsort()[::-1]
	#write results
	csvWriter = csv.writer(open(fname, 'w'))
	csvWriter.writerow(['Ground truth','prediction','probability'])
	count  = 0
	for i in ranked_idx:
		if rs[i] == y_test[i]:
			continue
		if count >= sample_n:
			break
		try:
			csvWriter.writerow([y_test[i] ,rs[i] ,str(probs[i])])
			count += 1
		except:
			pass


def get_wiki_curated_data(wiki_final_features_labels,wiki_final_features_freq100_300):
	raw_labels = pickle.load(open(wiki_final_features_labels,'rb'))
	label_ct = Counter(raw_labels)
	train_idx = []
	for idx,label in enumerate(raw_labels):
		if label_ct[label] >= 100:
			train_idx.append(idx)
	y = [raw_labels[i].lower() for i in train_idx]
	X = pickle.load(open(wiki_final_features_freq100_300,'rb'))
	return X,y

def get_wiki_bow_data(wiki_bow_freq100_300,wiki_bow_labels):
	X = pickle.load(open(wiki_bow_freq100_300,'rb'))
	#y = pickle.load(open(wiki_bow_label_ids,'rb'))
	labels = pickle.load(open(wiki_bow_labels,'rb'))
	label_ct = Counter(labels)
	train_idx = []
	for idx,label in enumerate(labels):
		if label_ct[label] >= 100:
			train_idx.append(idx)
	return X[train_idx], [labels[i].lower() for i in train_idx]

def get_wiki_bow_DWdata():
	X,y = get_wiki_bow_data()
	train_idx = []
	NDW_idx = []
	dd = Dictionary()
	for idx,label in enumerate(y):
		if label == '':
			continue
		if dd.isDW(label):
			train_idx.append(idx)
		else:
			NDW_idx.append(idx)
	NDW_y = [y[i] for i in NDW_idx]
	NDW_X = X[NDW_idx]
	y = [y[i] for i in train_idx]
	X = X[train_idx]
	return X,y,NDW_X,NDW_y

def get_wiki_curated_DWdata():
	X,y = get_wiki_curated_data()
	train_idx = []
	dd = Dictionary()
	for idx,label in enumerate(y):
		if label == '':
			continue
		if dd.isDW(label):
			train_idx.append(idx)
	y = [y[i] for i in train_idx]
	X = X[train_idx]
	return X,y


def get_wiki_curated_NDWdata():
	X,y = get_wiki_curated_data()
	train_idx = []
	dd = Dictionary()
	for idx,label in enumerate(y):
		if label == '':
			continue
		if not dd.isDW(label):
			train_idx.append(idx)
	y = [y[i] for i in train_idx]
	X = X[train_idx]
	return X,y

########################################## evaluation ##########################################
def eval_wiki_curated_DW(filter_col = 0):
	X,y = get_wiki_curated_DWdata()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3,test_size=0.7, random_state=42)
	clf = RandomForestClassifier(n_estimators=10,n_jobs=3)
	clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	preds = []
	iters = int(X_test.shape[0]/10000)
	for i in range(iters+1):
		preds.append(clf.predict(X_test[i*10000:(i*10000+10000)]))
	rs = np.concatenate(preds)
	macro = precision_recall_fscore_support(y_test, rs, average='macro')
	micro = precision_recall_fscore_support(y_test, rs, average='micro')
	topn_acc = [topn_accuracy(clf,X_test,y_test,topn = i+1) for i in range(10)]
	rs_dict = dict()
	rs_dict['method'] = 'curated'
	rs_dict['dataset'] = 'wikiDW'
	rs_dict['topn accuracy'] = topn_acc
	rs_dict['macro'] = macro
	rs_dict['micro'] = micro
	pickle.dump(rs_dict,open('/home/zhc415/myspace/govdata/wikiDW_curated_rs','wb'))
	generate_csv_results(clf,X_test,y_test,'/home/zhc415/Dropbox/repo/research/dataEngine/wikiDW_curated.csv',sample_n = 2000)


def eval_wiki_bow_DW(filter_col = 0):
	X,y,NDW_X,NDW_y = get_wiki_bow_DWdata()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3,test_size=0.7, random_state=42)
	clf = RandomForestClassifier(n_estimators=10,n_jobs=3)
	clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	preds = []
	iters = int(X_test.shape[0]/10000)
	for i in range(iters+1):
		preds.append(clf.predict(X_test[i*10000:(i*10000+10000)]))
	rs = np.concatenate(preds)
	macro = precision_recall_fscore_support(y_test, rs, average='macro')
	micro = precision_recall_fscore_support(y_test, rs, average='micro')
	topn_acc = [topn_accuracy(clf,X_test,y_test,topn = i+1) for i in range(10)]
	rs_dict = dict()
	rs_dict['method'] = 'bow'
	rs_dict['dataset'] = 'wikiDW'
	rs_dict['topn accuracy'] = topn_acc
	rs_dict['macro'] = macro
	rs_dict['micro'] = micro
	pickle.dump(rs_dict,open('/home/zhc415/myspace/govdata/wikiDW_bow_rs','wb'))
	generate_csv_results(clf,X_test,y_test,'/home/zhc415/Dropbox/repo/research/dataEngine/wikiDW_bow.csv',sample_n = 2000)
	generate_csv_results(clf,NDW_X,NDW_y,'/home/zhc415/Dropbox/repo/research/dataEngine/wikiDW_bow.csv',sample_n = 2000)




if __name__ == '__main__':
	#extract_wiki_features()
	#all_col_features = pickle.load(open(p_wiki_col_features,'rb'))
	#col_f_ids = feature2vec(all_col_features)
	pass



####################################### curated features #######################################


'''
col_f_matrix = pickle.load(open(wiki_final_features_sparse,'rb'))
X = preprocess_col_features(col_f_matrix)
labeled_X = col_f_matrix[train_idx,:] 
X = preprocess_col_features(labeled_X)
f_ids = pickle.load(open(wiki_final_features_ids,'rb'))
'''

# get rid of columns with non ascii features



# first select high quality labels 
# data selection  col freq > 100 ?
raw_labels = pickle.load(open(wiki_final_features_labels,'rb'))

#labels = pickle.load(open(wiki_final_features_label_ids,'rb'))
label_ct = Counter(raw_labels)
train_idx = []
for idx,label in enumerate(raw_labels):
	if label_ct[label] >= 100:
		train_idx.append(idx)




y = [raw_labels[i].lower() for i in train_idx]

X = pickle.load(open(wiki_final_features_freq100_300,'rb'))

col_dict = tagged_columns()
train_idx = []
for idx,label in enumerate(y):
	if label in col_dict:
		train_idx.append(idx)
		y[idx] = col_dict[label]


y = [y[i] for i in train_idx]
X = X[train_idx,:]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3,test_size=0.7, random_state=42)


clf = RandomForestClassifier(n_estimators=10,n_jobs=3)
clf.fit(X_train,y_train)
#clf.score(X_test,y_test)
preds = []
iters = int(X_test.shape[0]/10000)
for i in range(iters+1):
	preds.append(clf.predict(X_test[i*10000:(i*10000+10000)]))

rs = np.concatenate(preds)
macro = precision_recall_fscore_support(y_test, rs, average='macro')
micro = precision_recall_fscore_support(y_test, rs, average='micro')
topn_acc = [topn_accuracy(clf,X_test,y_test,topn = i+1) for i in range(10)]

rs_dict = dict()
rs_dict['method'] = 'curated'
rs_dict['dataset'] = 'wikitop'
rs_dict['topn accuracy'] = topn_acc
rs_dict['macro'] = macro
rs_dict['micro'] = micro
pickle.dump(rs_dict,open('/home/zhc415/myspace/govdata/wikiDW_curated_rs','wb'))



target_y = clf.predict(X_test)
probs = clf.predict_proba(X_test)
probs = [max(each) for each in probs]
probs = np.array(probs) 
ranked_idx = probs.argsort()[::-1]


theta = 0.5
csvWriter = csv.writer(open('/home/zhc415/Dropbox/repo/research/dataEngine/wikitop_curated.csv', 'w'))
csvWriter.writerow(['Ground truth','prediction','probability'])
for i in ranked_idx:
	if y_test[i] == target_y[i] or probs[i] < theta:
		continue
	csvWriter.writerow([y_test[i].encode('utf-8') , target_y[i].encode('utf-8') , probs[i]])


###################################### Bow #######################################

X = pickle.load(open(wiki_bow_freq100_300,'rb'))

#y = pickle.load(open(wiki_bow_label_ids,'rb'))
labels = pickle.load(open(wiki_bow_labels,'rb'))



label_ct = Counter(labels)


col_dict = tagged_columns()
train_idx = []
for idx,label in enumerate(labels):
	if label in col_dict and label_ct[label] >= 100:
		train_idx.append(idx)
		labels[idx] = col_dict[label]



X_train, X_test, y_train, y_test = train_test_split(X[train_idx], [labels[i] for i in train_idx], train_size=0.7,random_state=42)


clf = RandomForestClassifier(n_estimators=10,n_jobs=3)
clf.fit(X_train,y_train)

preds = []
iters = int(X_test.shape[0]/10000)
for i in range(iters+1):
	preds.append(clf.predict(X_test[i*10000:(i*10000+10000)]))

rs = np.concatenate(preds)
macro = precision_recall_fscore_support(y_test, rs, average='macro')
micro = precision_recall_fscore_support(y_test, rs, average='micro')
topn_acc = [topn_accuracy(clf,X_test,y_test,topn = i+1) for i in range(10)]

rs_dict = dict()
rs_dict['method'] = 'bow'
rs_dict['dataset'] = 'wiki'
rs_dict['topn accuracy'] = topn_acc
rs_dict['macro'] = macro
rs_dict['micro'] = micro
pickle.dump(rs_dict,open('/home/zhc415/myspace/govdata/wikitop_bow_rs','wb'))

target_y = clf.predict(X_test)
probs = clf.predict_proba(X_test)
probs = [max(each) for each in probs]
probs = np.array(probs) 
ranked_idx = probs.argsort()[::-1]


theta = 0.5
csvWriter = csv.writer(open('/home/zhc415/Dropbox/repo/research/dataEngine/wikitop_bow.csv', 'w'))
csvWriter.writerow(['Ground truth','prediction','probability'])
for i in ranked_idx:
	if y_test[i] == target_y[i] or probs[i] < theta:
		continue
	csvWriter.writerow([y_test[i].encode('utf-8') , target_y[i].encode('utf-8') , probs[i]])





trans = pickle.load(open(wiki_col_label_transormer,'rb'))




generate_Human_report(clf,X_test,y_test,index_dict,trans,'/home/zhc415/Dropbox/repo/research/dataEngine/wiki_bow.csv' ,5000,0.5)