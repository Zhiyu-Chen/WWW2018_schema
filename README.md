# Pipeline
## 1. Data Crawling

run 

'''python save.py
'''

to crawl datasets. (modify related path name in save.py and metadata.py)

Each dataset is a resource(gov_data.resource) which containing multiple csv files(gov_data.dataFile). 

## 2. Feature Extraction


Feature extraction has two steps:
	1. represent datasets in a list of resource objects (gov_data.resource).
	2. 1st pass to generate feature dictinary (preprocess.extract_gov_fdict)
	3. 2nd pass to generate histogram features, which depend on global information (preprocess.extract_gov_curated_features)

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
