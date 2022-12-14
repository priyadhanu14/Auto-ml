import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    
    data = None
    def __init__(self,data_dir):
            self.data_dir = data_dir
    
    def preprocess(self,test_size):

            #Loading datasets
            self.data=pd.read_csv(self.data_dir)

            #pre-processing
            #self.data.drop('Data point',inplace=True,axis=1)
            #self.data.drop('Time (s)',inplace=True,axis=1)

            # Removing Empty Records
            self.data.dropna(axis=0,how='all',inplace=True)

            # Removal of NaN Columns
            columns = self.data.columns
            nan_columns = list(filter(lambda x:self.data[x].isna().sum()>0.5*len(self.data),columns))
            self.data = self.data.drop(nan_columns,axis=1)

            # Correlation (Feature Selection)
            corr = self.data.corr()
            columns = corr.columns
            threshold = 0.1 # columns lesser than 0.01 and greater -0.01  not that correlated
            less_important_columns = []
            
            for column in columns:
                column_correlation = corr[column]
                not_correlated_count = 0
                # Checking correlation factor with other columns
                for other_column in columns:
                    if(column_correlation[other_column]<threshold and column_correlation[other_column]>-threshold):
                        # Counting no of columns with which the current column is less correlated
                        not_correlated_count+=1
              
            # Adding to the list if its less correlated with more than 50% of the total columns
            if(not_correlated_count>0.5*len(columns)):less_important_columns.append(column)
            
            self.data = self.data.drop(less_important_columns,axis=1)

            # Filling Values
            
            # lst = ['Feature 1','Label','Feature 4', 'Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11','Feature 12']
            # for wrd in lst:self.data[wrd]=self.data[wrd].interpolate()
            # self.data['Feature 3']=self.data['Feature 3'].ffill()
            
            # Finding Categorical and Continuous
            categorical_columns  = list(filter(lambda x:1.*self.data[x].nunique()/self.data[x].count() < 0.05,self.data.columns))
            continuous_columns   = list(set(self.data.columns).difference(set(categorical_columns)))

            for col in categorical_columns:self.data[col] = self.data[col].ffill()
            for col in continuous_columns:self.data[col] = self.data[col].interpolate()

            self.data=self.data.drop(1,axis=0)
            self.data=self.data.drop(0,axis=0)
            
            # Splitting x and y
            x=np.array(self.data.iloc[0:,0:11])
            y=np.array(self.data.iloc[0:,11])

            # data feature scaling
            scaler = StandardScaler()
            scaler = scaler.fit(x)
            x=scaler.transform(x)

            #Train test split
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=test_size,random_state=42)  
            
            return [X_train,y_train,X_val,y_val]