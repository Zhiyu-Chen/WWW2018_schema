import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from metadata import *
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from nltk.corpus import brown
from nltk.tokenize import ToktokTokenizer
from collections import Counter
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

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


def select_DW_columns(labels):
	wl = set()
	for each in brown.words():
		each = each.lower()
		if each.isalpha() and (each not in wl):
				wl.add(each)
	DW_labels = []
	DW_idx = []
	toktok = ToktokTokenizer()
	for idx,label in enumerate(labels):
		tokens = toktok.tokenize(label)
		flag = True
		for token in tokens:
			if token.isdigit():
				continue
			elif token.lower() not in wl:
				flag = False
				break
		if flag:
			DW_labels.append(label)
			DW_idx.append(idx)
	return DW_idx,DW_labels


def select_gov_ND_cols(all_resources):
	headers = []
	for rs in all_resources:
		for df in rs.data_files:
			header = df.header
			if not header:
				continue
			headers.extend(header)
	return headers


def tagged_columns(fname = 'std.csv',topn=2000):
	f = open(fname,'r')
	col_names = f.readline()
	col_names = col_names.split(',')
	col_dict = dict()
	count = 0
	for line in f:
		if count >= topn:
			break
		count += 1
		tokens = line.split(',')
		if 'meaningless' in tokens[6]:
			continue
		schema_label  =  tokens[1]
		taged_label = schema_label
		if tokens[3] != '': # normalized
			taged_label = tokens[3]
		col_dict[schema_label] = taged_label
	f.close()
	return col_dict


