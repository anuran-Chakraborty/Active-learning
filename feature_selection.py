import pandas as pd
import glob
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

def select_and_write(filename):
	print(filename)
	# Read dataset
	outfile='../featureSelected/'+filename
	nummFeatures=10


	df=pd.read_csv('../MicroarrayData/'+filename)
	y=df['class']
	df.drop('class',axis=1,inplace=True)
	X=df

	# perform feature selection
	selector=SelectKBest(score_func=mutual_info_classif,k=10)
	X=selector.fit_transform(X,y)
	X=pd.DataFrame(X)
	X['class']=y


	# write to csv
	X.to_csv(outfile,index=False)

# get list of files
os.chdir('MicroarrayData/')
filelist=glob.glob('*')

for files in filelist:
	select_and_write(files)