# -*- coding: utf-8 -*-
#-----cluster and regression-----



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans
import pandas as pd2
import numpy as np
import os
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
import time
import joblib


def get_data():
    path = os.getcwd()
    cluster_data = pd.read_csv(r"{}\gdsc_265drugs.csv".format(path),
                               index_col=0)
    cluster_data = cluster_data
    cluster_data.fillna(cluster_data.mean(),inplace=True)
    regression_data = pd.read_csv(r"{}\gdsc_265drugs.csv".format(path),
                                  index_col=0)
    regression_data = regression_data
    regression_data.fillna(regression_data.mean(),inplace=True)
    return cluster_data,regression_data


def cluster(cluster_data):
    cluster_result = {"drug":[],
                      "threshold_1":[],
                      "threshold_2":[]}
    for column in cluster_data.columns:
        series =  cluster_data[column].loc[pd.notnull(cluster_data[column])]
        cluster = KMeans(n_clusters=2, max_iter=1000).fit(series.values.reshape(-1,1))
        after_cluster = pd.concat([cluster_data[column],pd.Series(cluster.labels_,
                                                                  index=series.index,
                                                                  name="type")],axis=1)
        cluster_group = after_cluster.groupby("type")
        possible_list = []
        for label,items in cluster_group:
            possible_list.append(items[column].min())
            possible_list.append(items[column].max())
            
        possible_list.sort()
        threshold_1 = possible_list[1]
        threshold_2 = possible_list[2]
        
        cluster_result["drug"].append(column)
        cluster_result["threshold_1"].append(threshold_1)
        cluster_result["threshold_2"].append(threshold_2)
        
    cluster_result_df = pd.DataFrame(cluster_result)
    
    return cluster_result_df

def get_independent_data():
    
    path = os.getcwd()
    cluster_data,medicine = get_data()
    # cluster_result_df = cluster(cluster_data)
    data_1 = pd.read_csv(r"{}\cnv_fullgdsc.csv".format(path),index_col=0)
    data_1.fillna(0,inplace=True)
    #544 or 2134
    data_2 = pd.read_csv(r"{}\crispr_fullgdsc_544.csv".format(path),index_col=0)
    data_2.fillna(data_2.mean(),inplace=True)
    data_2 = data_2.apply(lambda x: (x-x.mean())/x.std(),axis=0)
    data_3 = pd.read_csv(r"{}\meth_fullgdsc.csv".format(path),index_col=0)
    data_3.fillna(data_3.mean(),inplace=True)
    data_3 = data_3.apply(lambda x: (x-x.mean())/x.std(),axis=0)
    data_4 = pd.read_csv(r"{}\mRNA_fullgdsc.csv".format(path),index_col=0)
    data_4.fillna(data_4.mean(),inplace=True)
    data_4 = data_4.apply(lambda x: (x-x.mean())/x.std(),axis=0)
    data_5 = pd.read_csv(r"{}\mutation_fullgdsc.csv".format(path),index_col=0)
    data_5.fillna(0,inplace=True)
    
    # threshold = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\pan_omics\drug_pan_threshold.xlsx")
    threshold = pd.read_excel("{}\drug_ID_threshold.xlsx".format(path),index_col=0)
    # threshold = cluster_result_df.loc[:,["drug","threshold_1"]]
    y_dummy = pd.DataFrame(index = medicine.index,
                           columns = medicine.columns)
    for cellline in medicine.index:
        for column in medicine.columns:
            threshold_value = threshold.loc[threshold["drug"] == column,"threshold_1"].values[0]
            if medicine[column][cellline] > threshold_value:
                y_dummy[column][cellline] = 0 
            else:
                y_dummy[column][cellline] = 1
    return data_1,data_2,data_3,data_4,data_5,medicine,threshold,y_dummy

class Drug_response:
    
    def __init__(self, data_1, data_2, data_3, data_4, data_5, medicine, threshold, y_dummy):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.data_4 = data_4
        self.data_5 = data_5
        self.y_values = medicine
        self.threshold = threshold
        self.y_dummy = y_dummy
        self.best_models = pd.Series(index=self.y_values.columns)
        self.predict_values = pd.DataFrame(columns = self.y_values.columns,
                                               index = self.y_values.index)
        self.predict_dummy = pd.DataFrame(columns = self.y_values.columns,
                                               index = self.y_values.index)

    def feature_selected(self,method="spearman"):
        correlated_result = {"medi":[],
                             "data_num":[],
                             "gen_name":[],
                             "score":[]}
        
        for medi in self.y_values.columns:
            print("feature selection of data 1 and {}".format(medi))
            data_1_score = []
            data_1_gene = []
            if len(self.data_1[self.data_1.columns[0]].unique()) < 5: #判断是否离散
                for gene in self.data_1.columns:
                    if len(self.data_1[gene].unique()) == 1:
                        continue
                    x = pd.Series(self.data_1[gene],name="x")
                    y = pd.Series(self.y_values[medi],name="y")
                    model = anova_lm(ols('y~C(x)',data=pd.concat([x,y],axis=1)).fit())
                    F_statistic = model["F"][0]
                    data_1_score.append(F_statistic)
                    data_1_gene.append(gene)
            else:
                for gene in self.data_1.columns:
                    correl = abs(self.y_values[medi].corr(self.data_1[gene],method=method))
                    data_1_score.append(correl)
                    data_1_gene.append(gene)
            data_1_score_mean = np.array(data_1_score).mean()
            data_1_score_std = np.array(data_1_score).std()
            for i in range(len(data_1_score)):
                if data_1_score[i] >= data_1_score_mean + 2*data_1_score_std:
                    correlated_result["medi"].append(medi)
                    correlated_result["data_num"].append(1)
                    correlated_result["gen_name"].append(data_1_gene[i])
                    correlated_result["score"].append(data_1_score[i])
                
            
            print("feature selection of data 2 and {}".format(medi))
            data_2_score = []
            data_2_gene = []
            if len(self.data_2[self.data_2.columns[0]].unique()) < 5: #判断是否离散
                for gene in self.data_2.columns:
                    if len(self.data_2[gene].unique()) == 1:
                        continue
                    x = pd.Series(self.data_2[gene],name="x")
                    y = pd.Series(self.y_values[medi],name="y")
                    model = anova_lm(ols('y~C(x)',data=pd.concat([x,y],axis=1)).fit())
                    F_statistic = model["F"][0]
                    data_2_score.append(F_statistic)
                    data_2_gene.append(gene)
            else:
                for gene in self.data_2.columns:
                    correl = abs(self.y_values[medi].corr(self.data_2[gene],method=method))
                    data_2_score.append(correl)
                    data_2_gene.append(gene)
            data_2_score_mean = np.array(data_2_score).mean()
            data_2_score_std = np.array(data_2_score).std()
            for i in range(len(data_2_score)):
                if data_2_score[i] >= data_2_score_mean + 2*data_2_score_std:
                    correlated_result["medi"].append(medi)
                    correlated_result["data_num"].append(2)
                    correlated_result["gen_name"].append(data_2_gene[i])
                    correlated_result["score"].append(data_2_score[i])
        
            print("feature selection of data 3 and {}".format(medi))
            data_3_score = []
            data_3_gene = []
            if len(self.data_3[self.data_3.columns[0]].unique()) < 5: #判断是否离散
                for gene in self.data_3.columns:
                    if len(self.data_3[gene].unique()) == 1:
                        continue
                    x = pd.Series(self.data_3[gene],name="x")
                    y = pd.Series(self.y_values[medi],name="y")
                    model = anova_lm(ols('y~C(x)',data=pd.concat([x,y],axis=1)).fit())
                    F_statistic = model["F"][0]
                    data_3_score.append(F_statistic)
                    data_3_gene.append(gene)
            else:
                for gene in self.data_3.columns:
                    correl = abs(self.y_values[medi].corr(self.data_3[gene],method=method))
                    data_3_score.append(correl)
                    data_3_gene.append(gene)
            data_3_score_mean = np.array(data_3_score).mean()
            data_3_score_std = np.array(data_3_score).std()
            for i in range(len(data_3_score)):
                if data_3_score[i] >= data_3_score_mean + 2*data_3_score_std:
                    correlated_result["medi"].append(medi)
                    correlated_result["data_num"].append(3)
                    correlated_result["gen_name"].append(data_3_gene[i])
                    correlated_result["score"].append(data_3_score[i])
                        
            print("feature selection of data 4 and {}".format(medi))
            data_4_score = []
            data_4_gene = []
            if len(self.data_4[self.data_4.columns[0]].unique()) < 5: #判断是否离散
                for gene in self.data_4.columns:
                    if len(self.data_4[gene].unique()) == 1:
                        continue
                    x = pd.Series(self.data_4[gene],name="x")
                    y = pd.Series(self.y_values[medi],name="y")
                    model = anova_lm(ols('y~C(x)',data=pd.concat([x,y],axis=1)).fit())
                    F_statistic = model["F"][0]
                    data_4_score.append(F_statistic)
                    data_4_gene.append(gene)
            else:
                for gene in self.data_4.columns:
                    correl = abs(self.y_values[medi].corr(self.data_4[gene],method=method))
                    data_4_score.append(correl)
                    data_4_gene.append(gene)
            data_4_score_mean = np.array(data_4_score).mean()
            data_4_score_std = np.array(data_4_score).std()
            for i in range(len(data_4_score)):
                if data_4_score[i] >= data_4_score_mean + 2*data_4_score_std:
                    correlated_result["medi"].append(medi)
                    correlated_result["data_num"].append(4)
                    correlated_result["gen_name"].append(data_4_gene[i])
                    correlated_result["score"].append(data_4_score[i])
        
            print("feature selection of data 5 and {}".format(medi))
            data_5_score = []
            data_5_gene = []
            if len(self.data_5[self.data_5.columns[0]].unique()) < 5: #判断是否离散
                for gene in self.data_5.columns:
                    if len(self.data_5[gene].unique()) == 1:
                        continue
                    x = pd.Series(self.data_5[gene],name="x")
                    y = pd.Series(self.y_values[medi],name="y")
                    model = anova_lm(ols('y~C(x)',data=pd.concat([x,y],axis=1)).fit())
                    F_statistic = model["F"][0]
                    data_5_score.append(F_statistic)
                    data_5_gene.append(gene)
            else:
                for gene in self.data_5.columns:
                    correl = abs(self.y_values[medi].corr(self.data_5[gene],method=method))
                    data_5_score.append(correl)
                    data_5_gene.append(gene)
            data_5_score_mean = np.array(data_5_score).mean()
            data_5_score_std = np.array(data_5_score).std()
            for i in range(len(data_5_score)):
                if data_5_score[i] >= data_5_score_mean + 2*data_5_score_std:
                    correlated_result["medi"].append(medi)
                    correlated_result["data_num"].append(5)
                    correlated_result["gen_name"].append(data_5_gene[i])
                    correlated_result["score"].append(data_5_score[i])
        
        self.selected_features = pd.DataFrame(correlated_result)
    
    def Run_Model_On_Row(self, Series, Model, param_grid, Include_data_1=1,Include_data_2=1,
                  Include_data_3=1,Include_data_4=1,Include_data_5=1):
        
        After_selected = pd.DataFrame()
        for_select = self.selected_features.loc[self.selected_features.medi==Series.name,:]
        for_select_1 = for_select.loc[for_select.data_num==1,:]
        for_select_2 = for_select.loc[for_select.data_num==2,:]
        for_select_3 = for_select.loc[for_select.data_num==3,:]
        for_select_4 = for_select.loc[for_select.data_num==4,:]
        for_select_5 = for_select.loc[for_select.data_num==5,:]
        
        print("processing the model of {}".format(Series.name))
        if Include_data_1 == 1:
            if len(for_select_1.index) > 0:
                for gene in for_select_1.gen_name:
                    After_selected = pd.concat([After_selected,
                                            self.data_1[gene]],axis=1)
        if Include_data_2 == 1:
            if len(for_select_2.index) > 0:
                for gene in for_select_2.gen_name:
                    After_selected = pd.concat([After_selected,
                                            self.data_2[gene]],axis=1)
        if Include_data_3 == 1:
            if len(for_select_3.index) > 0:
                for gene in for_select_3.gen_name:
                    After_selected = pd.concat([After_selected,
                                            self.data_3[gene]],axis=1)
        if Include_data_4 == 1:
            if len(for_select_4.index) > 0:
                for gene in for_select_4.gen_name:
                    After_selected = pd.concat([After_selected,
                                            self.data_4[gene]],axis=1)
        if Include_data_5 == 1:
            if len(for_select_5.index) > 0:
                for gene in for_select_5.gen_name:
                    After_selected = pd.concat([After_selected,
                                            self.data_5[gene]],axis=1)
        print("Start to run the model")
        threshold_value = self.threshold.loc[self.threshold["drug"]==Series.name,"threshold_1"].values[0]
        y = self.y_values[Series.name]
        X = After_selected
        combined = pd.concat([X,y],axis=1)
        grid = GridSearchCV(Model, param_grid, cv=5)
        grid.fit(X,y)
        best_model = grid.best_estimator_
        self.best_models[Series.name] = best_model
        for cellline in X.index:
            wait_for_pre_X = X.iloc[X.index == cellline,:]
            X_train = X.iloc[X.index != cellline, :]
            y_train = combined.iloc[X.index != cellline, -1]
            best_model.fit(X_train, y_train)
            joblib.dump(best_model,r"D:\wpk\1219\models\{}.sav".format(Series.name))
            predict_value = best_model.predict(wait_for_pre_X)
            
            self.predict_values[Series.name][cellline] = predict_value[0]
            
            if predict_value[0] > threshold_value:
                self.predict_dummy[Series.name][cellline] = 0
            else:
                self.predict_dummy[Series.name][cellline] = 1
    
    def Run_Model(self, Model, param_grid, Include_data_1=1,Include_data_2=1,
                  Include_data_3=1,Include_data_4=1,Include_data_5=1):
        self.y_values.apply(self.Run_Model_On_Row,
                            args=(Model,param_grid,Include_data_1,Include_data_2,
                                  Include_data_3,Include_data_4,Include_data_5),
                            axis=0)
    
    def score(self):
        pp = 0
        pn = 0
        nnp = 0
        nn = 0
        correct = 0
        self.confuse_matrix = pd.DataFrame(index=["pre_p","pre_n"],
                                           columns=["actual_p","actual_n"])
        
        final_result = {"medicine":[],
                        "accurancy":[],
                        "precision":[],
                        "specificity":[],
                        "sensitivity":[]}
        
        for medi in self.predict_dummy.columns:
            medi_correct = 0
            medi_pp = 0
            medi_pn = 0
            medi_np = 0
            medi_nn = 0
            for cellline in self.predict_dummy.index:
                pre_y_dummy = self.predict_dummy[medi][cellline]
                actual_y_dummy = self.y_dummy[medi][cellline]
                if pre_y_dummy == actual_y_dummy:
                    correct += 1
                    medi_correct += 1
                    if actual_y_dummy == 1:
                        pp += 1
                        medi_pp += 1
                    else:
                        nn += 1
                        medi_nn += 1
                else:
                    if actual_y_dummy == 1:
                        nnp += 1
                        medi_np += 1
                    else:
                        pn += 1
                        medi_pn += 1
            final_result["medicine"].append(medi)
            final_result["accurancy"].append(medi_correct/len(self.y_values.index))
            if medi_pp+medi_np == 0:
                final_result["sensitivity"].append(0)
            else:
                final_result["sensitivity"].append(medi_pp/(medi_pp+medi_np))
            if medi_pp+medi_pn == 0:
                final_result["precision"].append(0)
            else:
                final_result["precision"].append(medi_pp/(medi_pp+medi_pn))
            if medi_nn+medi_pn == 0:
                final_result["specificity"].append(0)
            else:
                final_result["specificity"].append(medi_nn/(medi_nn+medi_pn))
            
        self.final_df = pd.DataFrame(final_result)
        self.confuse_matrix.loc["pre_p","actual_p"] = pp
        self.confuse_matrix.loc["pre_p","actual_n"] = pn
        self.confuse_matrix.loc["pre_n","actual_p"] = nnp
        self.confuse_matrix.loc["pre_n","actual_n"] = nn
        print("accuracy is ",correct/(len(self.predict_dummy.index)*len(self.predict_dummy.columns)))
    
    
class Drug_response_new(Drug_response):
    
    def __init__(self,selected_features,data_1,data_2,data_3,data_4,data_5,medicine,threshold,y_dummy):
        Drug_response.__init__(self,data_1,data_2,data_3,data_4,data_5,
                               medicine,threshold,y_dummy)
        self.selected_features = selected_features

data_1,data_2,data_3,data_4,data_5,medicine,threshold,y_dummy = get_independent_data()

drug_response = Drug_response(data_1, data_2, data_3, data_4, data_5,
                              medicine,threshold,y_dummy)
drug_response.feature_selected(method="pearson")
Features = drug_response.selected_features
# Features = pd.read_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\挑选的变量_2134.xlsx",index_col=0)
# medicine.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\实际的y_values.xlsx")
# y_dummy.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\实际的y_dummy.xlsx")
# Features.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\挑选的变量_544.xlsx")
# threshold.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\threshold.xlsx")
# Features = pd.read_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\挑选的变量_544.xlsx",index_col=0)
# data_1_correlation = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\correlation_new.xlsx",
#                                    sheet_name="Sheet1",index_col=0)["correlation_1"]
# data_2_correlation = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\correlation_new.xlsx",
#                                    sheet_name="Sheet2",index_col=0)["correlation_2"]
# data_3_correlation = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\correlation_new.xlsx",
#                                    sheet_name="Sheet3",index_col=0)["correlation_3"]
# data_4_correlation = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\correlation_new.xlsx",
#                                    sheet_name="Sheet4",index_col=0)["correlation_4"]
# data_5_correlation = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\correlation_new.xlsx",
#                                    sheet_name="Sheet5",index_col=0)["correlation_5"]
# Features = pd.read_excel(r"D:\wpk\1113_pan\挑选的变量_gdsc_1std.xlsx")


drug_response_new = Drug_response_new(Features, data_1, data_2, 
                                      data_3, data_4, data_5, medicine,
                                      threshold,y_dummy)
param_grid = [
        {'kernel':['rbf'],'C':[1,5,10,30,50],'gamma':[0.0001,0.001,0.01,0.1,1]}
        ]
drug_response_new.Run_Model(SVR(), param_grid,Include_data_1=1, Include_data_2=1,
                            Include_data_3=1,Include_data_4=1,Include_data_5=1)
drug_response_new.score()
final_df = drug_response_new.final_df    #the predict results
predict_values = drug_response_new.predict_values
predict_dummy = drug_response_new.predict_dummy
model_params = drug_response_new.best_models


final_df.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\final_df_SVR_534_noMutation.xlsx")
predict_values.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\predict_values_SVR_534_noMutation.xlsx")
predict_dummy.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\predict_dummy_SVR_534_noMutation.xlsx")
model_params.to_excel(r"D:\wpk\1214\pan_gdsc2_2crispr\result\model_params_SVR_534_noMutation.xlsx")




