import pandas as pd
import numpy as np 
from math import sqrt

# Import necessary modules
from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from keras.models import Sequential
from keras.layers import Dense

from IPython.display import display

class Regression:
    X_train,X_val,X_test1,X_test2 = None,None,None,None
    y_train,y_val,y_test1,y_test2 = None,None,None,None

    RMSE_TEST1  = {'LINEAR_REGRESSION':0,'LINEAR_REGRESSION_SKLEARN':0,'DECISION_TREE':0,'RANDOM_FOREST':0,'NEURAL NETWORK':0}
    RMSE_TEST2  = {'LINEAR_REGRESSION':0,'LINEAR_REGRESSION_SKLEARN':0,'DECISION_TREE':0,'RANDOM_FOREST':0,'NEURAL NETWORK':0}

    MSE_TEST1  = {'LINEAR_REGRESSION':0,'LINEAR_REGRESSION_SKLEARN':0,'DECISION_TREE':0,'RANDOM_FOREST':0,'NEURAL NETWORK':0}
    MSE_TEST2  = {'LINEAR_REGRESSION':0,'LINEAR_REGRESSION_SKLEARN':0,'DECISION_TREE':0,'RANDOM_FOREST':0,'NEURAL NETWORK':0}


    def __init__(self,X_train,y_train,X_val,y_val,X_test1,y_test1,X_test2,y_test2):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test1 = X_test1
        self.X_test2 = X_test2
        self.y_train = y_train
        self.y_val = y_val
        self.y_test1 = y_test1
        self.y_test2 = y_test2

    def rmse(self,predictions,target):return sqrt(((predictions-target)**2).mean())

    def tabulate(self,options=['all']):
        # options = ['LINEAR_REGRESSION','LINEAR_REGRESSION_SKLEARN']
        E_TEST_lst = []
        if(options[0] == 'all'):E_TEST_lst = [self.RMSE_TEST1,self.RMSE_TEST2,self.MSE_TEST1,self.MSE_TEST2]
        else:
            rmse_test1 = {}
            rmse_test2 = {}
            mse_test1 = {}
            mse_test2 = {}

            lf = list(filter(lambda x:x in self.RMSE_TEST1.keys,options))
            if(len(lf)!=len(options)):
                print("Invalid Options Present")
                return

            for option in options:
                rmse_test1[option] = self.RMSE_TEST1[option]
                rmse_test2[option] = self.RMSE_TEST2[option]
                mse_test1[option] = self.MSE_TEST1[option]
                mse_test2[option] = self.MSE_TEST2[option]

            E_test_lst = [rmse_test1,rmse_test2,mse_test1,mse_test2]

        E_TEST_df = pd.DataFrame(E_TEST_lst,index=['TEST1','TEST2','TEST3','TEST4'])
        print(E_TEST_df)
        return E_TEST_df
        display(E_TEST_df)
        
    def model(self,model_type = 'all'):

        if(model_type == 'all'):
            print("starting")
            self.linear_regression()
            print("lin done")
            self.linear_regression_sklearn()
            self.decision_tree_sklearn()
            print("dec done")
            self.random_forest_sklearn()
            print("random done")
            self.tabulate(['all'])

        if(model_type == 'linear_regression'):
            self.linear_regression()
            self.tabulate(['LINEAR_REGRESSION'])

        elif(model_type == 'linear_regression_sklearn'):
            self.linear_regression_sklearn()
            self.tabulate(['LINEAR_REGRESSION_SKLEARN'])

        elif(model_type == 'decision_tree_sklearn'):
            self.decision_tree_sklearn()
            self.tabulate(['DECISION_TREE SKLEARN'])

        elif(model_type == 'random_forest_sklearn'):
            self.random_forest_sklearn()
            self.tabulate(['RANDOM_FOREST SKLEARN'])
            
    def linear_regression(self):
        # LINEAR_REGRESSION code
        costs_train=[]
        costs_val=[]
        cost_list=[]
        m=self.X_train.shape[0]
        ones= np.ones((m,1))
        self.X_train =np.concatenate((ones,self.X_train),axis=1)
        self.X_val = np.concatenate((np.ones((self.X_val.shape[0],1)),self.X_val),axis=1)
        self.X_test1 = np.concatenate((np.ones((self.X_test1.shape[0],1)),self.X_test1),axis=1)
        self.X_test2 = np.concatenate((np.ones((self.X_test2.shape[0],1)),self.X_test2),axis=1)
        n=self.X_train.shape[1]
        def cost(data,y,params):
            total_cost =0
            for i in range(len(data)):
                total_cost+=((1/(2*m))* ((data[i]*params).sum() -y[i])**2)
            cost_list.append(total_cost)
            #print(cost_list)
            return total_cost
        # gradient descent
        def grad_des(data,y,params,alpha,no_of_iterations):
            costs_array=[]
            for i in range(no_of_iterations):
                slopes = np.zeros (n)
                for j in range(len(data)):
                    for k in range (n):
                        slopes[k] += (1/m)*((data[j]*params).sum() -y[j])*data[j][k]
                params = params - (alpha*slopes)
                costs_array.append(cost(data,y,params))
            #costs.append(costs_array[-1])
            return [params,costs_array[-1]]
        
        sizes =[]
        params = np.zeros(n)
        costs_train=[]
        costs_val=[]
        costs_test=[]
        costs_test2=[]
        
        for i in range(0,len(self.X_train),100):
            params_1=grad_des(self.X_train[0:i],self.y_train[0:i],params,0.1,100)
            costs_train.append(params_1[1])
            sizes.append(i)
            y_predval = np.dot(params_1[0],self.X_val.T)
            costs_val.append((mean_squared_error(self.y_val,y_predval)))
            y_predtest1 = np.dot(params_1[0],self.X_test1.T)
            costs_test.append((mean_squared_error(self.y_test1,y_predtest1)))
            y_predtest2 = np.dot(params_1[0],self.X_test2.T)
            costs_test2.append((mean_squared_error(self.y_test2,y_predtest2)))

        self.RMSE_TEST1['LINEAR_REGRESSION'] = self.rmse(y_predtest1,self.y_test1)
        self.RMSE_TEST2['LINEAR_REGRESSION'] = self.rmse(y_predtest2,self.y_test2)

        self.MSE_TEST1['LINEAR_REGRESSION'] = mean_squared_error(y_predtest1,self.y_test1)
        self.MSE_TEST2['LINEAR_REGRESSION'] = mean_squared_error(y_predtest2,self.y_test2)
        
    def linear_regression_sklearn(self):
        model  = LinearRegression().fit(self.X_train,self.y_train)
        y_predtest1 = model.predict(self.X_test1)
        y_predtest2 = model.predict(self.X_test2)
 
        self.RMSE_TEST1['LINEAR_REGRESSION_SKLEARN'] = self.rmse(y_predtest1,self.y_test1)
        self.RMSE_TEST2['LINEAR_REGRESSION_SKLEARN'] = self.rmse(y_predtest2,self.y_test2)

        self.MSE_TEST1['LINEAR_REGRESSION_SKLEARN'] = mean_squared_error(y_predtest1,self.y_test1)
        self.MSE_TEST2['LINEAR_REGRESSION_SKLEARN'] = mean_squared_error(y_predtest2,self.y_test2)
              
    def decision_tree_sklearn(self):
        model = DecisionTreeRegressor(min_samples_leaf=5,random_state = 0).fit(self.X_train,self.y_train)
        y_predtest1 = model.predict(self.X_test1)
        y_predtest2 = model.predict(self.X_test2)
 
        self.RMSE_TEST1['DECISION_TREE'] = self.rmse(y_predtest1,self.y_test1)
        self.RMSE_TEST2['DECISION_TREE'] = self.rmse(y_predtest2,self.y_test2)

        self.MSE_TEST1['DECISION_TREE'] = mean_squared_error(y_predtest1,self.y_test1)
        self.MSE_TEST2['DECISION_TREE'] = mean_squared_error(y_predtest2,self.y_test2)
         
    def random_forest_sklearn(self):
        model = RandomForestRegressor(n_estimators =1000,random_state = 42).fit(self.X_train,self.y_train)
        y_predtest1 = model.predict(self.X_test1)
        y_predtest2 = model.predict(self.X_test2)
 
        self.RMSE_TEST1['RANDOM_FOREST'] = self.rmse(y_predtest1,self.y_test1)
        self.RMSE_TEST2['RANDOM_FOREST'] = self.rmse(y_predtest2,self.y_test2)

        self.MSE_TEST1['RANDOM_FOREST'] = mean_squared_error(y_predtest1,self.y_test1)
        self.MSE_TEST2['RANDOM_FOREST'] = mean_squared_error(y_predtest2,self.y_test2)
    
    def neural_network(self):

        # Model
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        model.fit(self.X_train, self.y_train,validation_data=(self.X_val,self.y_val), epochs=100, batch_size=12)
        y_predtest1=model.predict(self.X_test1)
        y_predtest2=model.predict(self.X_test2)

        self.RMSE_TEST1['NEURAL NETWORK'] = self.rmse(y_predtest1,self.y_test1)
        self.RMSE_TEST2['NEURAL NETWORK'] = self.rmse(y_predtest2,self.y_test2)

        self.MSE_TEST1['NEURAL NETWORK'] = mean_squared_error(y_predtest1,self.y_test1)
        self.MSE_TEST2['NEURAL NETWORK'] = mean_squared_error(y_predtest2,self.y_test2)
