import pandas as pd

# Import necessary modules
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import Dense

# confusion_matrix(actual, predicted)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# cm_display.plot()
# plt.savefig('confusion.png')

class Classification:

    X_train,X_val,X_test1,X_test2 = None,None,None,None
    y_train,y_val,y_test1,y_test2 = None,None,None,None

    RMSE_TEST1  = {'LINEAR REGRESSION':0,'LINEAR REGRESSION SKLEARN':0,'DECISION TREE':0,'RANDOM FOREST':0,'NEURAL NETWORK':0}
    RMSE_TEST2  = {'LINEAR REGRESSION':0,'LINEAR REGRESSION SKLEARN':0,'DECISION TREE':0,'RANDOM FOREST':0,'NEURAL NETWORK':0}


    ACC_1 = {'DECISION TREE':None,'LOGISTIC_REGRESSION':None,'NAIVE BAYES':None,'NEURAL NETWORK':None}
    ACC_2 = {'DECISION TREE':None,'LOGISTIC_REGRESSION':None,'NAIVE BAYES':None,'NEURAL NETWORK':None}
    
    def __init__(self,X_train,y_train,X_val,y_val,X_test1,y_test1,X_test2,y_test2):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test1 = X_test1
        self.X_test2 = X_test2
        self.y_train = y_train
        self.y_val = y_val
        self.y_test1 = y_test1
        self.y_test2 = y_test2

    def tabulate(self,options=['all']):
        E_TEST_lst = []
        if(options[0] == 'all'):E_TEST_lst = [self.ACC_1,self.ACC_2]
        else:
            ACC_test1 = {}
            ACC_test2 = {}

        # lf = list(filter(lambda x:x in list(self.RMSE_TEST1.keys()),options))
        # if(len(lf)!=len(options)):
        #     print("Invalid Options Present")
        #     return

        # for option in options:
        #     ACC_test1[option] = self.ACC_1[option]
        #     ACC_test2[option] = self.ACC_2[option]

        # E_test_lst = [ACC_test1,ACC_test2]

        E_TEST_df = pd.DataFrame(E_TEST_lst,index=['TEST1','TEST2'])
        print(E_TEST_df)
        return E_TEST_df


    def model(self,model_type = 'all'):
        if(model_type == 'all'):
            self.logistic_regression_sklearn()
            self.decision_tree_sklearn()
            self.naive_bayes_sklearn()
            # self.tabulate(['all'])

        elif(model_type == 'logistic_regression'):
            self.logistic_regression_sklearn()
            self.tabulate(['LOGISTIC_REGRESSION'])

        elif(model_type == 'decision_tree'):
            self.decision_tree_sklearn()
            self.tabulate(['DECISION TREE'])

        elif(model_type == 'naive_bayes'):
            self.naive_bayes_sklearn()
            self.tabulate(['NAIVE BAYES'])

        elif(model_type == 'neural_network'):
            self.neural_network()
            self.tabulate(['NEURAL NETWORK'])

    def logistic_regression_sklearn(self):
        clf = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        y_predtest1 = clf.predict(self.X_test1)
        y_predtest2 = clf.predict(self.X_test2)
        
        # test
        self.ACC_1['LOGISTIC_REGRESSION'] = accuracy_score(y_predtest1,self.y_test1)
        self.ACC_2['LOGISTIC_REGRESSION'] = accuracy_score(y_predtest2,self.y_test2)


    def generate_confusion_matrix(self,predictions,true_values):
        print(confusion_matrix(predictions,true_values))

    
    def decision_tree_sklearn(self):
        clf = DecisionTreeClassifier(random_state=0).fit(self.X_train,self.y_train)
        y_predtest1 = clf.predict(self.X_test1)
        y_predtest2 = clf.predict(self.X_test2)

        self.ACC_1['DECISION TREE'] = accuracy_score(y_predtest1,self.y_test1)
        self.ACC_2['DECISION TREE'] = accuracy_score(y_predtest2,self.y_test2)


    def naive_bayes_sklearn(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        y_predtest1 = gnb.predict(self.X_test1)
        y_predtest2 = gnb.predict(self.X_test2)

        self.ACC_1['NAIVE BAYES'] = accuracy_score(y_predtest1,self.y_test1)
        self.ACC_2['NAIVE BAYES'] = accuracy_score(y_predtest2,self.y_test2)

    def neural_network(self):

        # No of Outputs
        outputs = len(set(self.y_train))

        # Model Architecture
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(outputs,activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(self.X_train, self.y_train,validation_data=(self.X_val,self.y_val), epochs=100, batch_size=12)

        y_predtest1=model.predict(self.X_test1)
        y_predtest2=model.predict(self.X_test2)

        self.ACC_1['NEURAL NETWORK'] = accuracy_score(y_predtest1,self.y_test1)
        self.ACC_2['NEURAL NETWORK'] = accuracy_score(y_predtest2,self.y_test2)
