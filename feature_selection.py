import pandas as pd
import glob
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

def select_and_write(filename,num_feat):
	print(filename+' '+str(num_feat))
	# Read dataset
	outfile='../featureSelected/'+str(num_feat)+'/'+filename
	df=pd.read_csv('../MicroarrayData/'+filename)
	y=df['class']
	df.drop('class',axis=1,inplace=True)
	X=df

	# perform feature selection
	selector=SelectKBest(score_func=mutual_info_classif,k=num_feat)
	X=selector.fit_transform(X,y)
	X=pd.DataFrame(X)
	X['class']=y


	# write to csv
	X.to_csv(outfile,index=False)

num_features=[10,50,100]

# get list of files
os.chdir('MicroarrayData/')
filelist=glob.glob('*')

for nf in num_features:
	for files in filelist:
		select_and_write(files,nf)