# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 19:42:59 2022
"""
import pandas as pd
import os
import numpy as np
from scipy.stats import 
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
from sklearn.metrics import matthews_corrcoef

#All path can be revised
medicine = pd.read_csv(r"D:\wpk\0323\TCGA_model_data\Sorafenib01.csv")
medicine = medicine[["bcr_patient_barcode","measure_of_response"]]
#all the path can be reset accordingly
path_list = [r"D:\wpk\0323\TCGA_model_data\cnv_screen",
             r"D:\wpk\0323\TCGA_model_data\methy_screen",
             r"D:\wpk\0323\TCGA_model_data\mrna_screen",
             r"D:\wpk\0323\TCGA_model_data\mut_screen"]
for path in path_list:
    first_label = path.split("\\")[-1].split("_")[0]
    
    this_df = pd.DataFrame()
    for data_name in os.listdir(path):
        print(data_name)
        this_data = pd.read_csv(path+"\\"+data_name,index_col=0)
        if first_label == "methy":
            this_data.index = this_data["V1"]
            this_data = this_data.drop(["V1","Probe id","Related transcripts","gene"],axis=1)  
        if "TCGA" in this_data.index[0]: #correct format
            this_data = this_data.T
            if len(this_data.index[0].split("-")) > 3:
                this_data.index = pd.Series(this_data.index).apply(lambda x:
                                                                   "-".join(x.split("-")[:3]))
                this_data["code"] = this_data.index
                if len(this_df.index) == 0:
                    this_df = this_data
                else:
                    this_df = pd.merge(this_df,this_data,on="code",how="inner")

            else:
                this_data["code"] = this_data.index
                if len(this_df.index) == 0:
                    this_df = this_data
                else:
                    this_df = pd.merge(this_df,this_data,on="code",how="inner")
        else:
            if len(this_data.index[0].split("-")) > 3:
                this_data.index = pd.Series(this_data.index).apply(lambda x:
                                                                   "-".join(x.split("-")[:3]))
                this_data["code"] = this_data.index
                if len(this_df.index) == 0:
                    this_df = this_data
                else:
                    this_df = pd.merge(this_df,this_data,on="code",how="inner")
            else:
                this_data["code"] = this_data.index
                if len(this_df.index) == 0:
                    this_df = this_data
                else:
                    this_df = pd.merge(this_df,this_data,on="code",how="inner")
    this_df.index = this_df["code"]
    this_df.drop("code",axis=1,inplace=True)
    this_df.to_excel(r"D:\wpk\0323\TCGA_model_data\combined\\"+first_label+"_combined.xlsx")

                
def get_independent_data():
    data_1 = pd.read_excel(r"D:\wpk\0323\TCGA_model_data\combined\cnv_combined.xlsx",index_col=0)
    data_1.fillna(0,inplace=True)
    data_1 = data_1.T
    data_2 = pd.read_excel(r"D:\wpk\0323\TCGA_model_data\combined\methy_combined.xlsx",index_col=0)
    data_2.fillna(data_2.mean(),inplace=True)
    data_2 = data_2.apply(lambda x: (x-x.mean())/x.std(),axis=1)
    data_3 = pd.read_excel(r"D:\wpk\0323\TCGA_model_data\combined\mrna_combined2.xlsx",index_col=0)
    data_3.fillna(data_3.mean(),inplace=True)
    data_3 = data_3.apply(lambda x: (x-x.mean())/x.std(),axis=1)
    data_4 = pd.read_excel(r"D:\wpk\0323\TCGA_model_data\combined\mut_combined.xlsx",index_col=0)
    data_4.fillna(0,inplace=True)
    common_patient = pd.merge(pd.Series(data_1.columns,name="code"),
                              pd.Series(data_2.columns,name="code"),
                              how="inner")
    common_patient = pd.merge(common_patient,
                              pd.Series(data_3.columns,name="code"),
                              how="inner")
    common_patient = pd.merge(common_patient,
                              pd.Series(data_4.columns,name="code"),
                              how="inner")
    # threshold = pd.read_excel(r"C:\Users\Aubrey Chen\Desktop\pan_omics\drug_pan_threshold.xlsx")
    #threshold = cluster_result_df.loc[:,["drug","threshold_1"]]

    data_1 = data_1[common_patient.code].T
    data_2 = data_2[common_patient.code].T
    data_3 = data_3[common_patient.code].T
    data_4 = data_4[common_patient.code].T
    
    return data_1,data_2,data_3,data_4

def get_medicine(medi):
    medicine = pd.read_excel(r"D:\wpk\0323\TCGA_model_data\medicines\{}".format(medi))
    medicine = medicine[["bcr_patient_barcode","measure_of_response"]].drop_duplicates()
    return medicine

class Drug_response:
    
    def __init__(self, data_1, data_2, data_3, data_4, medicine, medi_name):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.data_4 = data_4
        self.y_values = medicine
        self.medi_name = medi_name
    
    def pre_process(self):
        common_patient = list(self.data_1.index)
        common_patient = pd.merge(pd.Series(common_patient,name="before"),
                                  pd.Series(self.y_values.bcr_patient_barcode,name="medi"),
                                  left_on="before",right_on="medi",
                                  how="inner")
        common_patient = list(common_patient.before)
        self.common_patient = common_patient
        self.data_1 = self.data_1.loc[common_patient,:].sort_index()
        self.data_2 = self.data_2.loc[common_patient,:].sort_index()
        self.data_3 = self.data_3.loc[common_patient,:].sort_index()
        self.data_4 = self.data_4.loc[common_patient,:].sort_index()
        self.y_values = self.y_values.loc[self.y_values.bcr_patient_barcode.isin(common_patient),:]
        self.y_values.index = self.y_values.bcr_patient_barcode
        self.y_values = self.y_values["measure_of_response"].sort_index()
    
    def feature_selected(self,method="spearman"):
        #
        correlated_result = {"medi":[],
                             "data_num":[],
                             "gen_name":[],
                             "score":[]}

        print("feature selection of data 2:{}".format(self.medi_name))
        data_1_medi = []
        data_1_score = []
        data_1_gene = []
        if len(self.data_1[self.data_1.columns[0]].unique()) < 5: #if discrete
            for gene in self.data_1.columns:
                if len(self.data_1[gene].unique()) == 1:
                    continue
                x = pd.Series(self.data_1[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                cont = pd.crosstab(x.values,y.values,rownames="x",colnames="y")
                stat,p,dof,expected = chi2_contingency(cont)
                data_1_score.append(stat)
                data_1_gene.append(gene)
                data_1_medi.append(self.medi_name)
        else:
            for gene in self.data_1.columns:
                x = pd.Series(self.data_1[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                y.index = x.index
                model = anova_lm(ols('x~C(y)',data=pd.concat([x,y],axis=1)).fit())
                F_statistic = model["F"][0]
                data_1_score.append(F_statistic)
                data_1_gene.append(gene)
                data_1_medi.append(self.medi_name)
        data_1_score_mean = np.array(data_1_score).mean()
        data_1_score_std = np.array(data_1_score).std()
        for i in range(len(data_1_score)):
            if data_1_score[i] >= data_1_score_mean + 2*data_1_score_std:
                correlated_result["medi"].append(self.medi_name)
                correlated_result["data_num"].append(1)
                correlated_result["gen_name"].append(data_1_gene[i])
                correlated_result["score"].append(data_1_score[i])
            
        
        print("feature selection of data 2:{}".format(self.medi_name))
        data_2_score = []
        data_2_gene = []
        data_2_medi = []
        if len(self.data_2[self.data_2.columns[0]].unique()) < 5: #判断是否离散
            for gene in self.data_2.columns:
                if len(self.data_2[gene].unique()) == 1:
                    continue
                x = pd.Series(self.data_2[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                cont = pd.crosstab(x.values,y.values,rownames="x",colnames="y")
                stat,p,dof,expected = chi2_contingency(cont)
                data_2_score.append(stat)
                data_2_gene.append(gene)
                data_2_medi.append(self.medi_name)
        else:
            for gene in self.data_2.columns:
                x = pd.Series(self.data_2[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                y.index = x.index
                model = anova_lm(ols('x~C(y)',data=pd.concat([x,y],axis=1)).fit())
                F_statistic = model["F"][0]
                data_2_score.append(F_statistic)
                data_2_gene.append(gene)
                data_2_medi.append(self.medi_name)
        data_2_score_mean = np.array(data_2_score).mean()
        data_2_score_std = np.array(data_2_score).std()
        for i in range(len(data_2_score)):
            if data_2_score[i] >= data_2_score_mean + 2*data_2_score_std:
                correlated_result["medi"].append(self.medi_name)
                correlated_result["data_num"].append(2)
                correlated_result["gen_name"].append(data_2_gene[i])
                correlated_result["score"].append(data_2_score[i])
    
        print("feature selection of data 3:{}".format(self.medi_name))
        data_3_score = []
        data_3_gene = []
        data_3_medi = []
        if len(self.data_3[self.data_3.columns[0]].unique()) < 5: #判断是否离散
            for gene in self.data_3.columns:
                if len(self.data_3[gene].unique()) == 1:
                    continue
                x = pd.Series(self.data_3[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                cont = pd.crosstab(x.values,y.values,rownames="x",colnames="y")
                stat,p,dof,expected = chi2_contingency(cont)
                data_3_score.append(stat)
                data_3_gene.append(gene)
                data_3_medi.append(self.medi_name)
        else:
            for gene in self.data_3.columns:
                x = pd.Series(self.data_3[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                y.index = x.index
                model = anova_lm(ols('x~C(y)',data=pd.concat([x,y],axis=1)).fit())
                F_statistic = model["F"][0]
                data_3_score.append(F_statistic)
                data_3_gene.append(gene)
                data_3_medi.append(self.medi_name)
        data_3_score_mean = np.array(data_3_score).mean()
        data_3_score_std = np.array(data_3_score).std()
        for i in range(len(data_3_score)):
            if data_3_score[i] >= data_3_score_mean + 2*data_3_score_std:
                correlated_result["medi"].append(self.medi_name)
                correlated_result["data_num"].append(3)
                correlated_result["gen_name"].append(data_3_gene[i])
                correlated_result["score"].append(data_3_score[i])
                    
        print("feature selection of data 4:{}".format(self.medi_name))
        data_4_score = []
        data_4_gene = []
        data_4_medi = []
        if len(self.data_4[self.data_4.columns[0]].unique()) < 5: #判断是否离散
            for gene in self.data_4.columns:
                if len(self.data_4[gene].unique()) == 1:
                    continue
                x = pd.Series(self.data_4[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                y.index = x.index
                cont = pd.crosstab(x.values,y.values,rownames="x",colnames="y")
                stat,p,dof,expected = chi2_contingency(cont)
                data_4_score.append(stat)
                data_4_gene.append(gene)
                data_4_medi.append(self.medi_name)
        else:
            for gene in self.data_4.columns:
                x = pd.Series(self.data_4[gene],name="x")
                y = pd.Series(self.y_values,name="y")
                model = anova_lm(ols('x~C(y)',data=pd.concat([x,y],axis=1)).fit())
                F_statistic = model["F"][0]
                data_4_score.append(F_statistic)
                data_4_gene.append(gene)
                data_4_medi.append(self.medi_name)
        data_4_score_mean = np.array(data_4_score).mean()
        data_4_score_std = np.array(data_4_score).std()
        for i in range(len(data_4_score)):
            if data_4_score[i] >= data_4_score_mean + 2*data_4_score_std:
                correlated_result["medi"].append(self.medi_name)
                correlated_result["data_num"].append(4)
                correlated_result["gen_name"].append(data_4_gene[i])
                correlated_result["score"].append(data_4_score[i])
                
        data_1_df = pd.DataFrame({"medi":data_1_medi,"gene":data_1_gene,"score":data_1_score})
        data_2_df = pd.DataFrame({"medi":data_2_medi,"gene":data_2_gene,"score":data_2_score})
        data_3_df = pd.DataFrame({"medi":data_3_medi,"gene":data_3_gene,"score":data_3_score})
        data_4_df = pd.DataFrame({"medi":data_4_medi,"gene":data_4_gene,"score":data_4_score})
        self.selected_features = pd.DataFrame(correlated_result)
        self.correlation = pd.concat([data_1_df,data_2_df,data_3_df,data_4_df])
    
    def Run_Model_On_Row(self, Model, param_grid, Include_data_1 = 1,
                         Include_data_2=1, Include_data_3=1,Include_data_4=1):
        
        After_selected = pd.DataFrame()
        for_select_1 = self.selected_features.loc[self.selected_features.data_num==1,:]
        for_select_2 = self.selected_features.loc[self.selected_features.data_num==2,:]
        for_select_3 = self.selected_features.loc[self.selected_features.data_num==3,:]
        for_select_4 = self.selected_features.loc[self.selected_features.data_num==4,:]

        After_selected.index = self.data_1.index
        print("we are processing the model of {}".format(self.medi_name))
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
        print("Start to run the model")
        
        y = self.y_values
        X = After_selected
        
        combined = pd.concat([X,y],axis=1)
        grid = GridSearchCV(Model, param_grid, cv=5)
        grid.fit(X,y)
        best_model = grid.best_estimator_
        joblib.dump(best_model,r"D:\wpk\0323\models\{}.sav".format(self.medi_name))
        self.best_model = best_model
        self.predict_values = []
        for cellline in X.index:
            wait_for_pre_X = X.iloc[X.index == cellline,:]
            X_train = X.iloc[X.index != cellline, :]
            y_train = combined.iloc[X.index != cellline, -1]
            best_model.fit(X_train, y_train)
            predict_value = best_model.predict(wait_for_pre_X)
            self.predict_values.append(predict_value[0])
        self.predict_values = np.array(self.predict_values)
        pre_y_label = KMeans(n_clusters=2, max_iter=1000).fit(self.predict_values.reshape(-1,1)).labels_
        if sum(pre_y_label == self.y_values) < len(pre_y_label)/2:
            pre_y_label = 1 - pre_y_label
        self.pre_y_label = pre_y_label           
    
    def score(self):
        
        final_result = {"medicine":[],
                        "accurancy":[],
                        "precision":[],
                        "specificity":[],
                        "sensitivity":[],
                        "MCC":[]}
        medi_correct = 0
        medi_pp = 0
        medi_pn = 0
        medi_np = 0
        medi_nn = 0
        pre_y_dummy = self.pre_y_label
        actual_y_dummy = self.y_values
        for idx in range(len(pre_y_dummy)):
            if pre_y_dummy[idx] == actual_y_dummy[idx]:
                medi_correct += 1
                if actual_y_dummy[idx] == 1:
                    medi_pp += 1
                else:
                    medi_nn += 1
            else:
                if actual_y_dummy[idx] == 1:
                    medi_np += 1
                else:
                    medi_pn += 1
        final_result["medicine"].append(self.medi_name)
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
        final_result["MCC"].append(matthews_corrcoef(self.y_values,self.pre_y_label))
        
        self.final_df = pd.DataFrame(final_result)
    
class Drug_response_new(Drug_response):
    
    def __init__(self,selected_features,data_1,data_2,data_3,data_4,medicine,medi_name):
        Drug_response.__init__(self,data_1,data_2,data_3,data_4,medicine,medi_name)
        self.selected_features = selected_features

data_1,data_2,data_3,data_4 = get_independent_data()
medicine_list = os.listdir(r"D:\wpk\0323\TCGA_model_data\medicines") #all medicine will be operated
Final_Features = pd.DataFrame()
Final_final_df = pd.DataFrame()
Final_sample_size = {"medi":[],
                     "sample_size":[]}
for medi in medicine_list:
    medi_name = medi.split(".")[0]
    print("processing {}".format(medi_name))
    medicine = get_medicine(medi)
    drug_response = Drug_response(data_1, data_2, data_3, data_4, medicine, medi_name=medi_name)
    drug_response.pre_process()
    drug_response.feature_selected()
    Features = drug_response.selected_features
    Final_Features = pd.concat([Final_Features,Features])
    this_all_correlation = drug_response.correlation
    this_all_correlation.to_csv(r"D:\wpk\0323\results\correlations\{}_score.csv".format(medi_name))
    drug_response_new = Drug_response_new(Features,data_1, data_2, data_3, data_4, medicine, medi_name=medi_name)
    drug_response_new.pre_process()
    
    param_grid = [
        {'kernel':['rbf'],'C':[1,5,10,30,50],'gamma':[0.0001,0.001,0.01,0.1,1]}
        ]
    drug_response_new.Run_Model_On_Row(SVR(), param_grid)
    this_sample_size = len(drug_response_new.pre_y_label)
    Final_sample_size["medi"].append(medi_name)
    Final_sample_size["sample_size"].append(this_sample_size)
    drug_response_new.score()
    pre_y_list = drug_response_new.pre_y_label
    actual_y_list = drug_response_new.y_values
    y_values = pd.concat([actual_y_list,pd.Series(pre_y_list,index=actual_y_list.index,
                                                  name="pre_y_values")],axis=1)
    #y_values.to_excel(r"D:\wpk\0323\results\pre_and_actual\{}_y_values.xlsx".format(medi_name))
    Final_final_df = pd.concat([Final_final_df,drug_response_new.final_df])

Final_Features.to_excel(r"D:\wpk\0323\results\selected_features.xlsx")
pd.DataFrame(Final_sample_size).to_excel(r"D:\wpk\0323\results\Sample_size.xlsx")
Final_final_df.to_excel(r"D:\wpk\0323\results\prediction_performance.xlsx")


test_y = drug_response_new.y_values
medicine.sort_values(by="bcr_patient_barcode")
medicine.index = medicine.bcr_patient_barcode
medicine.sort_index()

data_1_new = drug_response_new.data_1
data_1_new.drop_duplicates()
data_2_new = drug_response.data_2
yy = drug_response_new.y_values
# medicine.to_excel(r"D:\wpk\1219\ccle_pan2134\result\实际的y_values.xlsx")
data_1.drop_duplicates()
medicine.drop_duplicates()

  
