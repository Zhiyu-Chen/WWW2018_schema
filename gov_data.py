from __future__ import division

from glob import glob
import json
import csv
import codecs
import numpy as np
import math 
from collections import Counter
import os
from metadata import *
import pickle
import random

class dataFile(object):
	'''
	TODO:
	a lot of cases to handle the errors
	1. no such file file (due to download errors)
	2. have the file with data.csv, but is not csv file -> col errors
	'''
	def __init__(self,d_path):
		self.path = d_path
		self.read_data_meta()
		self.df_id = d_path.split('/')[-1]
		self.header = None
		self.status = False
		self.reason = None
		self.col_num = 0
		#self.row_num = 0
		self.data_size = 0
		#self.read_data_cols()

	def show_few_data(self, num_line = 3):
		try:
			f = open(self.path + '/data.csv','r')
			for i in range(num_line):
				print(f.readline())
		except:
			print("reading file error.")

	def get_raw_content(self):
		try:
			f = open(self.path + '/data.csv','r')
			content = f.read()
		except:
			content = ""
			print("reading file error.")	
		return content	

	def get_content_tokens(self):
		try:
			f = open(self.path + '/data.csv','r')
			for line in f:
				tokens = line.lower().strip().split(',')
				for token in tokens:
					if token.isalpha():
						content.add(token)
		except:
			print("reading file error.")	
		return content	


	def read_data_meta(self):
		data_meta = json.loads(open(self.path + '/meta.json','rb').readline().decode('utf-8'))
		self.name = data_meta['name']
		self.url = data_meta['url']


	def read_data_cols(self):
		try:
			self.data_size = os.path.getsize(self.path + '/data.csv')
			with codecs.open(self.path + '/data.csv', encoding="utf-8", errors="replace") as ifile:
				reader = csv.reader(ifile, delimiter=',', quotechar='"')
				for header in reader:
					self.header = header
					self.status = True
					self.col_num = len(self.header)
					break
			return True
		except:
			self.reason = "open error"
			return False


	def row_and_cell_count(self,cell_counter,normalize = False):
		self.row_num = 0
		try:
			with codecs.open(self.path + '/data.csv', encoding="utf-8", errors="replace") as ifile:
				reader = csv.reader(ifile, delimiter=',', quotechar='"')
				reader.next() #jump the header
				for row in reader:
					self.row_num += 1
					cell_counter.update(row)
			return cell_counter
		except:
			self.reason = "open error"
			return cell_counter
'''

	def dbpedia_annotate(self,confidence=0.20,support=10):	
		self.table_nes = []
		content = self.get_content_tokens()
		for each in content:
			try:
				self.table_nes.extend(spotlight.annotate('http://localhost:2222/rest/annotate',each, confidence,support))
			except:
				pass

'''

class resource(object):
	'''
	NOTICE:
	1. notes should go through preprocessing (containing HTML tags, etc)
	'''
	def __init__(self,r_path='/home/zhc415/myspace/exp/data_engine/automatic-eureka/data/fe7598fe-9189-45db-a6e8-3332867bc402'):
		self.path = r_path
		self.rs_id = r_path.split('/')[-1]
		self.status = False
		self.reason = "no csv"
		self.notes_len = 0
		self.data_num = 0
		self.data_type = []
		self.read_datasets()

	@staticmethod
	def r_notes_prep():
		pass
		

	def read_datasets(self):
		 resources = glob(self.path + '/*')
		 self.resources = list(filter(lambda x:x[-4:] != 'json',resources))
		 self.num_data = len(resources)
		 self.read_resource_meta()
		 self.data_files = []
		 # read all dataset columns
		 for rs in self.resources:
		 	df = dataFile(rs)
		 	if df.read_data_cols():
		 		self.status = True
		 		self.data_files.append(df)


	def read_resource_meta(self):
		'''
		read resource metadata 
		'''
		meta_path = self.path + '/meta.json'
		rs_meta = json.loads(open(meta_path,'rb').readline().decode('utf-8'))
		self.notes = rs_meta['notes']
		self.notes_len = len(self.notes.split())
		self.title = rs_meta['title']
		self.tags = []
		self.tags = [each['name'] for each in rs_meta['tags']]

	def dbpedia_annotate(self,confidence=0.20,support=10):	
		self.rs_nes = []
		for each in [self.notes,self.title]:
			try:
				self.rs_nes.extend(spotlight.annotate('http://localhost:2222/rest/annotate',each, confidence,support))
			except:
				pass


def resource_preprocessing(all_resources):
	'''
	1. wrong csv files
	2. normalize special characters
	'''
	resources = wrong_csv(all_resources)
	resources = normalize_headers(resources)
	return resources

def wrong_csv(resources):
	'''
	1. if line length <=1, false
	2. if line length >1, len(line1.splits) != len(line2.splits), false
	3. no headers ?
	'''
	for idx in range(len(resources)):
		rs_status = True 
		for df_idx in range(len(resources[idx].data_files)):
			f = open(resources[idx].data_files[df_idx].path + '/data.csv','r')
			try:
				line1 = f.readline()
				line2 = f.readline()
				line3 = f.readline()
			except:
				resources[idx].data_files[df_idx].status = False
				resources[idx].data_files[df_idx].reason = 'few lines'
				continue
			if len(line1.strip().split(',')) <=1:
				resources[idx].data_files[df_idx].status = False
				resources[idx].data_files[df_idx].reason = 'few columns'
				continue
			if len(line2.strip().split(',')) != len(line3.strip().split(',')):
				resources[idx].data_files[df_idx].status = False
				resources[idx].data_files[df_idx].reason = "not aligned"
				continue
			#Check false header:'COLLECTING TAXES  FULL DATA:  2012-13,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\r\n'
			false_header = False
			for each in line1.split(','):
				if len(each) == 0:
					false_header = True
					break
			if false_header:
				resources[idx].data_files[df_idx].status = False
				resources[idx].data_files[df_idx].reason = "false header"
				continue
			#good file should have false value
			rs_status = False
		if not rs_status:
			resources[idx].status = True
		else:
			resources[idx].status = False
	return resources

def normalize_headers(resources):
	'''

	'''
	for idx in range(len(resources)):
		for df_idx in range(len(resources[idx].data_files)):
			if not resources[idx].data_files[df_idx].status:
				continue
			for h_idx in range(len(resources[idx].data_files[df_idx].header)):
				resources[idx].data_files[df_idx].header[h_idx] = resources[idx].data_files[df_idx].header[h_idx].lower()
				chrs = list(resources[idx].data_files[df_idx].header[h_idx])
				for ch_idx in range(len(chrs)):
					if (not chrs[ch_idx].isalpha()) and (not chrs[ch_idx].isdigit()):
						chrs[ch_idx] = '_'
				resources[idx].data_files[df_idx].header[h_idx] = ''.join(chrs).lower()
	return resources





def read_resources(data_path = data_path):
	resources = glob(data_path +'*')
	all_resources = [resource(each) for each in resources]
	return all_resources

def get_federal_resources(all_resources):
	f_resources = []
	for rs in all_resources: 
		mp = rs.path + '/meta.json'   
		f = open(mp,'r').readlines()[0]    
		if 'programCode' in f:
			f_resources.append(rs)
	return f_resources

def restore_good_resources():
	return pickle.load(open(p_good_resources,'rb'))

def select_resources(all_resources,fsize = 10,rs_ct = 300):
	rs_indices = list(range(len(all_resources)))
	random.shuffle(rs_indices)
	rand_gov_resources = []
	for rs_idx in rs_indices:
		flag = True
		for each_data in all_resources[rs_idx].data_files:
			d_path = each_data.path + '/data.csv'
			if os.path.getsize(d_path)/1024/1024 > fsize:
				flag = False
				break
		if flag:
			rand_gov_resources.append(all_resources[rs_idx])
		if len(rand_gov_resources) >= rs_ct:
			break
	return rand_gov_resources


def pre_cell(cell,case_folding = True):
	'''
	preprocess cell before extracting features
	1. case folding
	2. replace numbers ?
	'''
	cell = cell.strip()
	if case_folding:
		cell = cell.lower()
	return cell

def get_col_features(col):
	'''
	extract the features from one column
	1st only collect features from 100 rows ? (not random distribution)
		1.avg cell length (excluding numeric features )
		2.number of characters
		3. % of numeric characters
		4. % of alphabetical characters
		5. % of symbolic chatacters (characters other than"A" to "Z', "a" to "z', and "0" to "9")
		6. Percentage of numeric cells. We define a cell containing only "0" to "9", "." and "%" a numeric cell.
		7. maximum value in the column
		8. minimum value in the column
	'''
	if len(col) == 0:
		return False
	feature_dict = dict()
	cell_len = []
	num_ct = 0
	letter_ct = 0
	symbol_ct = 0
	total = 0
	max_value = float('nan')
	min_value = float('nan')
	min_cell_len = 999999
	max_cell_len = 0
	numeric_ct = 0
	unigram = Counter() 
	col_hist = Counter()
	try:
		for cell in col:
			cell = pre_cell(cell)
			unigram.update([each for each in cell])
			total += 1
			col_hist.update([cell])
			if cell.replace('.','',1).isdigit():
				numeric_ct += 1
				cell_value = float(cell)
				if math.isnan(max_value):
					max_value = cell_value
				if math.isnan(min_value):
					min_value = cell_value
				if cell_value > max_value:
					max_value = cell_value
				if cell_value < min_value:
					min_value = cell_value
			cell_len.append(len(cell))
			for ch  in cell:
				if ch.isalpha():
					letter_ct += 1 
				elif ch.isdigit():
					num_ct += 1
				else:
					symbol_ct += 1

			if len(cell) > max_cell_len:
				max_cell_len = len(cell)
			elif len(cell) < min_cell_len:
				min_cell_len = len(cell)

		total_ch = np.sum(cell_len)
		feature_dict['avg_cell_len'] = np.mean(cell_len)  #avg cell length (excluding numeric features )
		feature_dict['total_ch'] = total_ch # number of characters
		if total_ch != 0:
			feature_dict['%number_ch'] = num_ct/total_ch # % of numeric characters
			feature_dict['%letter_ch'] = letter_ct/total_ch # % of alphabetical characters
			feature_dict['%sym_ch'] = symbol_ct/total_ch #% of symbolic chatacters (characters other than"A" to "Z', "a" to "z', and "0" to "9")
		else:
			feature_dict['%number_ch'] = 0
			feature_dict['%letter_ch'] = 0 # % of alphabetical characters
			feature_dict['%sym_ch'] = 0
		if total != 0:
			feature_dict['%num_cells'] = numeric_ct/total
		else:
			feature_dict['%num_cells'] = 0
		feature_dict['min_value'] = min_value # minimum value in the column
		feature_dict['max_value'] = max_value # maximum value in the column
		if sum(col_hist.values()) != 0:
			feature_dict['unique_ratio'] = len(col_hist)/sum(col_hist.values())
		else:
			feature_dict['unique_ratio'] =  0
		feature_dict['unigram'] = unigram
		feature_dict['hist'] = col_hist
		feature_dict['min_cell_len'] = min_cell_len
		feature_dict['max_cell_len'] = max_cell_len
	except:
		return False
	return feature_dict





def search_rs_headers(all_resources,header):
	resources = []
	for rs in all_resources:
		for df in rs.data_files:
			if  df.header and header in df.header:
				resources.append(rs)
				break
	return resources




def search_df_headers(all_resources,header):
	dfs = []
	for rs in all_resources:
		for df in rs.data_files:
			if  df.header:
				headers = [each.lower() for each in df.header]
				if  header in headers:
					dfs.append(df)
	return dfs