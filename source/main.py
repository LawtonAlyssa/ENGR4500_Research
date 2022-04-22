from ctypes import sizeof
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from time import sleep
from tqdm import tqdm
import random
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_linnerud, make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Functions
# ======================================================================

def getClassificationModelType(class_model_type, **kwargs):
    if class_model_type == "svm": return MultiOutputClassifier(svm.SVC(kernel='rbf', **kwargs)) # binary classification model
    if class_model_type == "random_forest": return RandomForestClassifier(max_depth=2, random_state=0, **kwargs) # binary classification model
    if class_model_type == "ann": return MultiOutputClassifier(MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 8), random_state=1, max_iter=10000, **kwargs))

def getColDataTypes(data_df, discrete_info_df):
    return [col for col in data_df if discrete_info_df[col]['discrete']], [col for col in data_df if not discrete_info_df[col]['discrete']]

def getEdgeData(data_df, cols):
    return data_df[cols]

def getHeartData():
    df = pd.read_csv("data/heart.csv")
    df.set_index(keys='ID', inplace=True)
    return df

def getHeartInfo():
    df = pd.read_csv("data/heart.info")
    df.set_index(keys='info', inplace=True)
    return df  

def getMeanSquaredError(y_pred_df, y_df):
    return round(mean_squared_error(y_pred=y_pred_df, y_true=y_df), 7)

def getModelAccuracy(y_pred_df, y_df):
    return accuracy_score(y_true=y_df, y_pred=y_pred_df)

def getRegressionModelType(reg_model_type, **kwargs):
    if reg_model_type == "ridge": return MultiOutputRegressor(Ridge(random_state=123, **kwargs))
    if reg_model_type == "random_forest": return RandomForestRegressor(max_depth=2, random_state=0, **kwargs)
    if reg_model_type == "k_neighbors": return KNeighborsRegressor(n_neighbors=2, **kwargs)
    if reg_model_type == "svr": return MultiOutputRegressor(LinearSVR(random_state=0, tol=1e-05, max_iter=100000, **kwargs))
    if reg_model_type == "ann": return MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1, max_iter=100000, **kwargs)

def getSampleData(data_df):
    # n = 12500
    # training: 10000
    # testing: 2500
    return data_df.sample(n=62500, random_state=random.randint(a=0, b=2e9))

def main():
    final_results_df = pd.DataFrame(columns=['model_type', 'model', 'accuracy', 'class_gen_model', 'reg_gen_model'])
    current_date_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    for _ in tqdm(range(1000)):
        sim_results_df = simulation_instance()
        # reorder columns
        final_results_df = pd.concat([final_results_df, sim_results_df], axis=0)
        final_results_df = final_results_df[['class_gen_model', 'reg_gen_model', 'model', 'accuracy']].convert_dtypes()
        final_results_df.to_csv('results/results_{}.csv'.format(current_date_time))      
    final_results_df = final_results_df.groupby(['model', 'class_gen_model', 'reg_gen_model'])['accuracy'].mean()
    final_results_df.to_csv('results/results_{}.csv'.format(current_date_time))      
    print()
    print(final_results_df)
    
def modelFit(model, X, y):  
    try:
        # print("Fitting model...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "")
            model.fit(X, y) 
    except Exception:
        print(traceback.print_exc)
        # print("Fitting model using ravel()...")
        print(y.ravel())
        model.fit(X, y.ravel())

def fitClassificationFeatures(X, y):
    # edge features 
    models = []
    y_dfs = []
    # model_data_type = 'classification'
    # print("Fitting", model_data_type, "models...")
    for model_name in ['svm', 'random_forest', 'ann']:
        y_temp = y
        # print("Fitting", model_name, "...")
        if model_name=='ann':
            model = getClassificationModelType(model_name)
        else:
            if model_name=='svm': # pseudo-classification
                model = getRegressionModelType('svr')
            elif model_name=='random_forest': # pseudo-classification
                model = getRegressionModelType(model_name)
            y_df = pd.DataFrame(y)
            y_dum_df = pd.get_dummies(y_df, columns=y.columns, prefix=y.columns)
            y = y_dum_df
        # print(y.head())
        y_dfs.append(y)
        modelFit(model, X, y) 
        models.append(model)
        y = y_temp
    # print("Finished edge features classification model fitting...")
    return models, y_dfs # fitted classfication models of edge features

def predictClassificationFeatures(models, X, y, discrete_cols, results_cols):
    results_df = pd.DataFrame()
    gen_cols = []
    
    # edge features generating discrete features
    model_data_type ='classification'
    # print("Predicting", model_data_type, "models...")
    model_names = ['svm', 'random_forest', 'ann']
    for i in range(len(model_names)):
        model_name = model_names[i]
        # print("Predicting", model_name, "...")
        model = models[i]
        # print(model) 
        y_cols = pd.get_dummies(y, columns=y.columns, prefix=y.columns).columns \
            if model_name=='svm' or model_name=='random_forest' else y.columns
        heart_gen_prime_df = pd.DataFrame(model.predict(X), columns=y_cols, index=y.index)
        
        if model_name=='svm' or model_name=='random_forest': # binary
            for y_col in discrete_cols:
                y_pred_cols = [y_pred_col for y_pred_col in heart_gen_prime_df.columns if y_pred_col.startswith(y_col+"_")]
                y_pred_cols_df = heart_gen_prime_df[y_pred_cols]
                y_pred_cols_df.columns = [y_pred_col.split(y_col+"_", 1)[1] for y_pred_col in y_pred_cols]
                heart_gen_prime_df[y_col] = y_pred_cols_df[y_pred_cols_df.columns].idxmax(axis=1)
                heart_gen_prime_df.drop(columns=y_pred_cols, inplace=True)                
        
        gen_cols.append(heart_gen_prime_df)
        
        # UNCOMMENT
        # print('expected')
        # print(y[discrete_cols].head(10))
        # print([len(y[col].unique()) for col in discrete_cols])

        # print('predicted')
        # print(heart_gen_prime_df.head(10))
        # print([len(heart_gen_prime_df[col].unique()) for col in heart_gen_prime_df.columns])
        
        if isinstance(heart_gen_prime_df, object): # and isinstance(y, np.int64):
            # print('convert y_pred_df int64')
            heart_gen_prime_df = heart_gen_prime_df.astype('int64')
        if isinstance(heart_gen_prime_df, np.int32) and isinstance(y, np.float64):
            # print('convert y_pred_df float')
            heart_gen_prime_df = heart_gen_prime_df.astype('float64')
        accuracy = [getModelAccuracy(y_pred_df=heart_gen_prime_df[col], y_df=y[col]) for col in y.columns]
        results_df = results_df.append(pd.DataFrame([model_data_type, model_name, accuracy]).transpose())
          
    results_df.reset_index(drop=True, inplace=True)
    results_df.columns = results_cols
    # print("gen_class_cols_results_df:")  
    # print(results_df)
    
    return gen_cols

def fitRegressionFeatures(X, y):
    # edge features 
    models = []
    y_dfs = []
    # model_data_type = 'regression'
    # print("Fitting", model_data_type, "models...")
    for model_name in ['ridge', 'random_forest', 'svr', 'ann']:
        # print("Fitting", model_name, "...")
        model = getRegressionModelType(model_name)
        y_dfs.append(y)
        modelFit(model, X, y) 
        models.append(model)
    # print("Finished edge features regression model fitting...")
    return models # fitted regression models of edge features

def predictRegressionFeatures(models, X, y, results_cols):
    results_df = pd.DataFrame()
    gen_cols = []
    
    # edge features generating continuous features
    model_data_type ='regression'
    # print("Predicting", model_data_type, "models...")
    model_names = ['ridge', 'random_forest', 'svr', 'ann']
    for i in range(len(model_names)):
        model_name = model_names[i]
        # print("Predicting", model_name, "...")
        model = models[i]
        heart_gen_prime_df = pd.DataFrame(model.predict(X), columns=y.columns, index=y.index)
        mse = [getMeanSquaredError(y_pred_df=heart_gen_prime_df[col], y_df=y[col]) for col in y.columns]
        results_df = results_df.append(pd.DataFrame([model_data_type, model_name, mse]).transpose())
        gen_cols.append(heart_gen_prime_df)
    results_df.reset_index(drop=True, inplace=True)
    results_df.columns = results_cols
    # print("gen_reg_cols_results_df:")  
    # print(results_df)
    
    return gen_cols
    
def fitAllFeatures(X, y):
    # all 13 features 
    models = []
    # model_data_type = 'classification'
    # print("Fitting", "models...")
    for model in ['svm', 'random_forest', 'ann']:
        # print("Fitting", model, "...")
        model = getClassificationModelType(model)
        modelFit(model, X, y) 
        models.append(model)
    # print("Finished all features classification model fitting...")
    return models # fitted classification models of all 13 features

def predictAllFeatures(models, X, y, results_cols):
    results_df = pd.DataFrame()
    
    # all 13 features
    model_data_type ='classification'
    # print("Predicting", model_data_type, "models...")
    model_names = ['svm', 'random_forest', 'ann']
    for i in range(len(model_names)):
        model_name = model_names[i]
        # print("Predicting", model_name, "...")
        model = models[i]
        y_prime_df = pd.DataFrame(model.predict(X), index=y.index)
        accuracy = getModelAccuracy(y_pred_df=y_prime_df, y_df=y)
        results_df = results_df.append(pd.DataFrame([model_data_type, model_name, accuracy]).transpose())
          
    results_df.reset_index(drop=True, inplace=True)
    results_df.columns = results_cols
    # print("results_df:")  
    # print(results_df)
    
    return results_df

def simulation_instance():
    heart_data_df = getSampleData(getHeartData())
    heart_label_df = pd.DataFrame(heart_data_df['class'])
    heart_info_df = getHeartInfo()
    
    for df in [heart_data_df, heart_info_df]: df.drop(columns=['class'], inplace=True)
    
    discrete_cols, continuous_cols = getColDataTypes(data_df=heart_data_df, discrete_info_df=heart_info_df)
    # print(discrete_cols, continuous_cols)
    heart_data_continuous_df = heart_data_df[continuous_cols]
    heart_data_discrete_df = heart_data_df[discrete_cols]
    
    # normalizes continuous features
    heart_data_continuous_df = (heart_data_continuous_df-heart_data_continuous_df.min())/(heart_data_continuous_df.max()-heart_data_continuous_df.min())

    # recombines normalized continuous features with regression features    
    heart_data_df = pd.concat([heart_data_continuous_df, heart_data_discrete_df], axis=1)   
    
    # splits data into training and testing dataframes
    X_heart_train_df, X_heart_test_df, y_heart_train_df, y_heart_test_df = train_test_split(heart_data_df, heart_label_df, test_size = 0.2, random_state=random.randint(a=0, b=2e9), shuffle=True)
        
    # fits on training data and all 13 features
    models_all_feat = fitAllFeatures(X=X_heart_train_df, y=y_heart_train_df)
        
    edge_cols = ['age', 
                 'sex', 
                 'resting_blood_pressure', 
                 'fasting_blood_sugar', 
                 'resting_electrocardiographic_results', 
                 'maximum_heart_rate_achieved', 
                 'exercise_induced_angina']
    
    # edge data collection
    heart_edge_train_df = getEdgeData(data_df=X_heart_train_df, cols=edge_cols)
    heart_edge_test_df = getEdgeData(data_df=X_heart_test_df, cols=edge_cols)
    
    # expected generated columns
    heart_gen_train_df = X_heart_train_df.drop(columns=edge_cols)
    heart_gen_test_df = X_heart_test_df.drop(columns=edge_cols)

    discrete_cols, continuous_cols = getColDataTypes(data_df=heart_gen_test_df, discrete_info_df=heart_info_df)
        
    y = heart_gen_train_df[discrete_cols]
    
    # generates discrete features using classification models
    models_class_feat_gen, y = fitClassificationFeatures(X=heart_edge_train_df, y=y)
    heart_gen_class_cols = predictClassificationFeatures(models=models_class_feat_gen, X=heart_edge_test_df, y=heart_gen_test_df[discrete_cols], discrete_cols=discrete_cols, results_cols=['model_type', 'model', 'accuracy'])
    
    # generates continuous features using regression models
    models_reg_feat_gen = fitRegressionFeatures(X=heart_edge_train_df, y=heart_gen_train_df[continuous_cols])
    heart_gen_reg_cols = predictRegressionFeatures(models=models_reg_feat_gen, X=heart_edge_test_df, y=heart_gen_test_df[continuous_cols], results_cols=['model_type', 'model', 'MSE'])
    
    # predict all 13 features using test data
    predictAllFeatures(models=models_all_feat, X=X_heart_test_df, y=y_heart_test_df, results_cols=['model_type', 'model', 'accuracy'])

    # predict all 13 features using edge features combined with generated columns from above
    simulation_result_df = pd.DataFrame(columns=['model_type', 'model', 'accuracy', 'class_gen_model', 'reg_gen_model'])
    
    for c in range(len(heart_gen_class_cols)):
        for r in range(len(heart_gen_reg_cols)):
            class_models = ['svm', 'random_forest', 'ann']
            reg_models = ['ridge', 'random_forest', 'svr', 'ann']
            
            X_heart_test_prime_df = pd.concat([heart_edge_test_df, heart_gen_class_cols[c], heart_gen_reg_cols[r]], axis=1)
            
            results_df = predictAllFeatures(models=models_all_feat, X=X_heart_test_prime_df, y=y_heart_test_df, results_cols=['model_type', 'model', 'accuracy'])
            results_df = pd.concat([results_df, pd.DataFrame([[class_models[c], reg_models[r]]]*3)], axis=1)
            results_df.columns = ['model_type', 'model', 'accuracy', 'class_gen_model', 'reg_gen_model']
            
            simulation_result_df = pd.concat([simulation_result_df, results_df], axis=0)
    return simulation_result_df

if __name__ == '__main__':
    print("Running main...")
    main()
    print("Program terminated")