import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import lightgbm as lgb
from datetime import datetime
from joblib import dump

class ClassifierTuner:
    def __init__(self,X,y,target_name,*,test_size=0.2,random_state=9587,n_jobs=3,path_model=''):
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5,random_state=self.random_state)
        self.results_df = None
        self.target_name = target_name.lower()
        self.n_jobs = n_jobs
        self.path_model = path_model
        self.anomesdia = datetime.now().strftime("%Y%m%d")

    def svc_objective(self, trial):
        try:
            C = trial.suggest_loguniform('C', 1e-5, 1e5)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            degree = trial.suggest_int('degree', 2, 10) if kernel == 'poly' else None
            gamma = trial.suggest_loguniform('gamma', 1e-5, 1e2)
            coef0 = trial.suggest_uniform('coef0', -1, 1)
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            tol = trial.suggest_loguniform('tol', 1e-6, 1e-2)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
            max_iter = trial.suggest_int('max_iter', 100, 1000, step = 50)
            decision_function_shape = trial.suggest_categorical('decision_function_shape', ['ovr', 'ovo'])
            random_state = trial.suggest_categorical('random_state', [self.random_state])

            model = Pipeline([('scaler', StandardScaler()),
                              ('svc', SVC(C=C,
                                          kernel=kernel,
                                          degree=degree,
                                          gamma=gamma,
                                          coef0=coef0,
                                          shrinking=shrinking,
                                          tol=tol,
                                          class_weight=class_weight,
                                          max_iter=max_iter,
                                          decision_function_shape=decision_function_shape,
                                          random_state=random_state))])

            scores = cross_val_score(model,self.X_train,self.y_train,cv=self.cv,scoring='accuracy',n_jobs=self.n_jobs)
            accuracy = np.mean(scores)

            return accuracy

        except Exception as e:
            print(f"An exception occurred in SVC objective function: {str(e)}")
            return 0.0  # Return a default value (e.g., 0.0) in case of an exception

    def knn_objective(self, trial):
        try:
            n_neighbors = trial.suggest_int('n_neighbors', 1, 1000)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            leaf_size = trial.suggest_int('leaf_size', 10, 100, step = 10)
            p = trial.suggest_categorical('p', [1, 2, np.inf])

            model = Pipeline([('scaler', StandardScaler()),
                              ('knn', KNeighborsClassifier(n_neighbors=n_neighbors,
                                                           weights=weights,
                                                           algorithm=algorithm,
                                                           leaf_size=leaf_size,
                                                           p=p))])

            scores = cross_val_score(model,self.X_train,self.y_train,cv=self.cv,scoring='accuracy',n_jobs=self.n_jobs)
            accuracy = np.mean(scores)

            return accuracy

        except Exception as e:
            print(f"An exception occurred in KNN objective function: {str(e)}")
            return 0.0  # Return a default value (e.g., 0.0) in case of an exception

    def lightgbm_objective(self, trial):
        try:
            params = {
                'boosting': trial.suggest_categorical('boosting', ['gbdt', 'rf', 'dart', 'goss']),
                'objective': trial.suggest_categorical('objective', ['binary', 'multiclass', 'ovr']),
                'num_class': len(np.unique(self.y_train)),
                'random_state': trial.suggest_categorical('random_state', [self.random_state]),
                'verbose': trial.suggest_categorical('verbose', [-1]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step = 10),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 2, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-10, 1e2),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-10, 1e2),
            }

            model = Pipeline([('scaler', StandardScaler()),('lgb', lgb.LGBMClassifier(**params))])

            scores = cross_val_score(model,self.X_train,self.y_train,cv=self.cv,scoring='accuracy',n_jobs=self.n_jobs)
            accuracy = np.mean(scores)

            return accuracy

        except Exception as e:
            print(f"An exception occurred in LightGBM objective function: {str(e)}")
            return 0.0  # Return a default value (e.g., 0.0) in case of an exception

    def optimize_models(self, n_trials=50):
        svc_study = optuna.create_study(direction='maximize')
        knn_study = optuna.create_study(direction='maximize')
        lgb_study = optuna.create_study(direction='maximize')
        
        # Optimize SVC
        t_initial = datetime.now()
        svc_study.optimize(self.svc_objective, n_trials=n_trials)
        best_svc_params = svc_study.best_params
        final_svc_model = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**best_svc_params))])
        final_svc_model.fit(self.X_train, self.y_train)
        t_final = datetime.now()
        train_time_svc = (t_final - t_initial).total_seconds()

        # Optimize KNN
        t_initial = datetime.now()
        knn_study.optimize(self.knn_objective, n_trials=n_trials)
        best_knn_params = knn_study.best_params
        final_knn_model = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(**best_knn_params))])
        final_knn_model.fit(self.X_train, self.y_train)
        t_final = datetime.now()
        train_time_knn = (t_final - t_initial).total_seconds()

        # Optimize LightGBM
        t_initial = datetime.now()
        lgb_study.optimize(self.lightgbm_objective, n_trials=n_trials)
        best_lgb_params = lgb_study.best_params
        final_lgb_model = Pipeline([('scaler', StandardScaler()), ('lgb', lgb.LGBMClassifier(**best_lgb_params))])
        final_lgb_model.fit(self.X_train, self.y_train)
        t_final = datetime.now()
        train_time_lgb = (t_final - t_initial).total_seconds()

        # Evaluate the final models on the test set
        y_pred_svc = final_svc_model.predict(self.X_test)
        y_pred_knn = final_knn_model.predict(self.X_test)
        y_pred_lgb = final_lgb_model.predict(self.X_test)

        accuracy_svc = accuracy_score(self.y_test, y_pred_svc)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        accuracy_lgb = accuracy_score(self.y_test, y_pred_lgb)

        precision_svc = precision_score(self.y_test, y_pred_svc, average='weighted')
        precision_knn = precision_score(self.y_test, y_pred_knn, average='weighted')
        precision_lgb = precision_score(self.y_test, y_pred_lgb, average='weighted')

        recall_svc = recall_score(self.y_test, y_pred_svc, average='weighted')
        recall_knn = recall_score(self.y_test, y_pred_knn, average='weighted')
        recall_lgb = recall_score(self.y_test, y_pred_lgb, average='weighted')

        f1_svc = f1_score(self.y_test, y_pred_svc, average='weighted')
        f1_knn = f1_score(self.y_test, y_pred_knn, average='weighted')
        f1_lgb = f1_score(self.y_test, y_pred_lgb, average='weighted')

        # Save the results in a DataFrame
        self.results_df = pd.DataFrame({
            'target': [self.target_name, self.target_name, self.target_name],
            'model': ['SVM', 'k-NN', 'LightGBM'],
            'accuracy': [accuracy_svc, accuracy_knn, accuracy_lgb],
            'precision': [precision_svc, precision_knn, precision_lgb],
            'recall': [recall_svc, recall_knn, recall_lgb],
            'f1_score': [f1_svc, f1_knn, f1_lgb],
            'train_time':[train_time_svc, train_time_knn, train_time_lgb],
            'best_params': [best_svc_params, best_knn_params, best_lgb_params],
            'y_test': [self.y_test, self.y_test, self.y_test],
            'y_pred': [y_pred_svc, y_pred_knn, y_pred_lgb]
        })

        # Save the best models using joblib
        dump(final_svc_model,f'{self.path_model}/{self.target_name}_svm_{self.anomesdia}.joblib')
        dump(final_knn_model,f'{self.path_model}/{self.target_name}_knn_{self.anomesdia}.joblib')
        dump(final_lgb_model,f'{self.path_model}/{self.target_name}_lgbm_{self.anomesdia}.joblib')

    def get_results(self):
        return self.results_df

# Example usage:
# classifier_tuner = ClassifierTuner(X, y)
# classifier_tuner.optimize_models(n_trials=50)
# results = classifier_tuner.get_results()
# print(results)
