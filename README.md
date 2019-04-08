# Pipeline
## 1. about the data 

### crawling data from data.gov
run 

	python save.py


to crawl datasets. (modify related path name in save.py and metadata.py)

Each dataset is a resource(gov_data.resource) which contains multiple csv files(gov_data.dataFile). 

### WikiTables

available at http://websail-fe.cs.northwestern.edu/TabEL/

## 2. Feature Extraction


Feature extraction for data.gov has three steps:

1. represent datasets in a list of resource objects (gov_data.resource).

2. 1st pass to generate feature dictionary (preprocess.extract_gov_fdict)

3. 2nd pass to generate histogram features, which depend on global information (preprocess.extract_gov_curated_features)

The processing for WikiTables is similar.

## 3. Training and Testing

example running in label_predict.py


# Reference

	@inproceedings{chen2018generating,
	  title={Generating Schema Labels through Dataset Content Analysis},
	  author={Chen, Zhiyu and Jia, Haiyan and Heflin, Jeff and Davison, Brian D},
	  booktitle={Companion of the The Web Conference 2018 on The Web Conference 2018},
	  pages={1515--1522},
	  year={2018},
	  organization={International World Wide Web Conferences Steering Committee}
	}
