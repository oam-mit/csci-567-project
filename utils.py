import pandas as pd
import numpy as np
from typing import List, Callable
import pickle
from sklearn.model_selection import train_test_split

class Mapping():
    def __init__(self, name, mapping):
        self.name = name
        self.mapping = mapping
        

class HousePriceData():

    #Maps for the datas which cannot be converted directly by .categories
    utilites_map = {"AllPub" : 0, "NoSewr": 1, "NoSeWa":2, "ELO":3}
    condition_map = {index:key for key, index in enumerate(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn'])}
    exterior_map = {index: key for key, index in enumerate(["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"])}
    masvnrtype_map = {index:key for key, index in enumerate(["BrkCmn", "BrkFace", "CBlock", "None", "Stone"])}
    exterior_quality_map = kitchen_qual_map = {index:key for key, index in enumerate(["Ex", "Gd", "TA", "Fa", "Po"])}
    bsmt_quality_map = bsmt_cond_map  = {index:key for key, index in enumerate(["Ex", "Gd", "TA", "Fa", "Po", "NA"])}
    functional_map = {index:key for key, index in enumerate(["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"])}
    pool_qc_map = {index:key for key, index in enumerate(["Ex", "Gd", "TA", "Fa", "NA"])}
    misc_feature_map = {index:key for key, index in enumerate(["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"])}
    sales_type_feature_map = {index:key for key, index in enumerate(["WD", "CWD", "VWD", "New", "COD", "Con", "COD", "Con", "ConLw" ,"ConLI", "ConLD", "Oth"])}
    msZoning_map  = {index: key for key, index in enumerate(["A", "C", "FV", "I", "RH", "RL", "RP", "RM"])}
    msSubclass_Map = {index: key for key, index in enumerate([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])}


    def __init__(self, path:str, x_preprocessors = None, y_preprocessors = None):
        self.__path = path
        self.__df = pd.read_csv(path, keep_default_na=False,na_values=["NA", np.nan, ""])

        #Correct spelling mistake
        self.__df["Exterior2nd"] = self.__df["Exterior2nd"].replace({"CmentBd": "CemntBd", "Wd Shng" :"WdShing", "Brk Cmn": "BrkComm"})
        self.__df["MSZoning"] = self.__df["MSZoning"].replace({"C (all)":"C"})
        self.__x_preprocessors = x_preprocessors
        self.__y_preprocessors = y_preprocessors



        self.__dropped_columns:List[str] = []

    
    def check_na(self, print_null_cols = True):
        null_cols = self.__df.columns[self.__df.isna().any()].tolist()
        if print_null_cols:
            for col in null_cols:
                print(col, sum(self.__df[col].isnull().tolist()))
        
        return null_cols
    
    def drop_column(self, col):
        self.__df = self.__df.drop(col, axis=1)
        self.__dropped_columns.append(col)


    def clean_data(self, null_cols:List[str], replace_by:List[str]):
        self.__df = self.__df.drop("Id", axis=1)
        
        for col, replace in zip(null_cols, replace_by):
            
            if replace == "M":
                mean = self.__df[col].loc[self.__df[col].notnull()].mean()
                self.__df[col] = self.__df[col].fillna(round(mean, 0))
            elif replace == "Z":
                self.__df[col] = self.__df[col].fillna(0.0)
            elif replace == "NA":
                self.__df[col] = self.__df[col].fillna("NA")
            elif replace == "None":
                self.__df[col] = self.__df[col].fillna("None")
            elif replace == "ZwM":
                mean = self.__df[col].loc[self.__df[col].notnull()].mean()
                self.__df[col] = self.__df[col].fillna(round(mean, 0))
                self.__df[col].loc[self.__df[col] == 0.0] = round(mean, 0)
            elif replace == "NonewNA":
                self.__df[col] = self.__df[col].fillna("NA")
                self.__df[col].loc[self.__df[col] == "None"] = "NA"
            else:
                self.__df[col] = self.__df[col].fillna(replace)
    


    def check_na_describe(self, null_cols, show_data = False):
        for col in null_cols:
            print("Values for column: ", col)
            if show_data:
                print(self.__df[col].loc[self.__df[col].isnull()].head(5))
                print(self.__df[col].loc[self.__df[col].notnull()].head(5))
            if self.__df[col].dtype == float:
                if len(self.__df[col].loc[self.__df[col] == 0.0].tolist()) != 0:
                    print("Zeros exist")
                else:
                    print("No Zeros in Data")
                mean = self.__df[col].mean()
                print("mean = ", mean)
            else:
                parameters = self.__df[col].unique().tolist()
                print("Unique parameters = ", parameters)
                
            print("\n")

    def convert_to_categorical(self) -> List[Mapping]:
        mapping:List[Mapping] = []


        category_list = ["Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "ExterCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "Fence", "SaleCondition","Neighborhood", "Foundation", "PavedDrive"]

        self.__df["MSSubClass"] = self.__df["MSSubClass"].replace(self.msSubclass_Map)
        mapping.append(Mapping("MSSubClass", self.msSubclass_Map))

        self.__df["MSZoning"] = self.__df["MSZoning"].replace(self.msZoning_map)
        mapping.append(Mapping("MSZoning", self.msZoning_map))

        self.__df["Utilities"] = self.__df["Utilities"].replace(self.utilites_map)
        mapping.append(Mapping("Utilities", self.utilites_map))

        self.__df["Condition1"] = self.__df["Condition1"].replace(self.condition_map)
        mapping.append(Mapping("Condition1", self.condition_map))

        self.__df["Condition2"] = self.__df["Condition2"].replace(self.condition_map)
        mapping.append(Mapping("Condition2", self.condition_map))


        self.__df["Exterior1st"] = self.__df["Exterior1st"].replace(self.exterior_map)
        mapping.append(Mapping("Exterior1st", self.exterior_map ))

        self.__df["Exterior2nd"] = self.__df["Exterior2nd"].replace(self.exterior_map)
        mapping.append(Mapping("Exterior2nd", self.exterior_map))

        self.__df["MasVnrType"] = self.__df["MasVnrType"].replace(self.masvnrtype_map)
        mapping.append(Mapping("MasVnrType", self.masvnrtype_map))
        
        self.__df["ExterQual"] = self.__df["ExterQual"].replace(self.exterior_quality_map)
        mapping.append(Mapping("ExterQual", self.exterior_quality_map))

        self.__df["BsmtQual"] = self.__df["BsmtQual"].replace(self.bsmt_quality_map)
        mapping.append(Mapping("BsmtQual", self.bsmt_quality_map))

        self.__df["BsmtCond"] = self.__df["BsmtCond"].replace(self.bsmt_cond_map)
        mapping.append(Mapping("BsmtCond", self.bsmt_cond_map))

        self.__df["KitchenQual"] = self.__df["KitchenQual"].replace(self.kitchen_qual_map)
        mapping.append(Mapping("KitchenQual", self.kitchen_qual_map))

        self.__df["Functional"] = self.__df["Functional"].replace(self.functional_map)
        mapping.append(Mapping("Functional", self.functional_map))

        self.__df["PoolQC"] = self.__df["PoolQC"].replace(self.pool_qc_map)
        mapping.append(Mapping("PoolQC", self.pool_qc_map))

        self.__df["MiscFeature"] = self.__df["MiscFeature"].replace(self.misc_feature_map)
        mapping.append(Mapping("MiscFeature", self.misc_feature_map))

        self.__df["SaleType"] = self.__df["SaleType"].replace(self.sales_type_feature_map)
        mapping.append(Mapping("SaleType", self.sales_type_feature_map))


        for category in category_list:
            self.__df[category] = self.__df[category].astype("category")
            category_mappings = {category_label: category_index for category_index, category_label in enumerate(self.__df[category].cat.categories)}
            mapping.append(Mapping(category, category_mappings))
            self.__df[category] = self.__df[category].cat.codes

        self.__mapping = mapping
        

        return mapping
    
    def get_dataframe(self):
        return self.__df
    
    def get_dropped_columns(self) ->List[str]:
        return self.__dropped_columns
    
    def fit_preprocessors(self):
        X = self.__df.drop("SalePrice", axis=1)
        Y = self.__df["SalePrice"]

        if self.__x_preprocessors:
            self.__x_preprocessors.fit(X)
        if self.__y_preprocessors:
            self.__y_preprocessors.fit(Y.values.reshape(-1, 1))
    
    def save_mapping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__mapping, f)
    
    def get_x_y(self):
        X = self.__df.drop("SalePrice", axis=1)
        Y = self.__df["SalePrice"]

        if self.__x_preprocessors:
            X = self.__x_preprocessors.transform(X)
        if self.__y_preprocessors:
            Y = self.__y_preprocessors.transform(Y.values.reshape(-1, 1))
        
        return X,Y
    
    def get_train_test_split(self, test_size=0.2):
        X,Y = self.get_x_y()

        return train_test_split(X, Y, test_size=test_size, random_state=42)
    
    def get_processors(self):
        return self.__x_preprocessors, self.__y_preprocessors
    
    def get_columns_to_drop(self, threshold):
        output_column = 'SalePrice'

        correlations = self.__df.corr(method="pearson")[output_column]

        selected_columns = correlations[(correlations <= threshold) & (correlations >= -threshold)].index.tolist()

        return selected_columns



class TestingData():
    def __init__(self, mapping_path:str, data_path, dropped_columns:List[str], x_preprocessor= None) -> None:
        self.__mapping_path = mapping_path
    
        with open(mapping_path, 'rb') as f:
            self.mappings:List[Mapping] = pickle.load(f)
        self.__df = pd.read_csv(data_path, keep_default_na=False,na_values=["NA", np.nan, ""])
        self.__dropped_columns = dropped_columns
        self.__na_columns = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
        self.ids = self.__df["Id"].to_list()
        self.__df = self.__df.drop("Id", axis=1)
        self.__x_preprocessor = x_preprocessor
    

    def preprocess_data(self):
    
        self.__df["Alley"] = self.__df["Alley"].fillna("NA")
        self.__df[self.__na_columns] = self.__df[self.__na_columns].fillna("NA")
        self.__df["MasVnrType"] = self.__df["MasVnrType"].fillna("None")
        self.__df["Exterior2nd"] = self.__df["Exterior2nd"].replace({"CmentBd": "CemntBd", "Wd Shng" :"WdShing", "Brk Cmn": "BrkComm"})
        self.__df["MSZoning"] = self.__df["MSZoning"].replace({"C (all)":"C"})
        most_frequent_values = self.__df.mode().iloc[0]
        self.__df = self.__df.fillna(most_frequent_values)
        for col in self.__dropped_columns:
            self.__df = self.__df.drop(col, axis=1)
        for mapping in self.mappings:
            if mapping.name not in self.__dropped_columns:
                self.__df[mapping.name] = self.__df[mapping.name].replace(mapping.mapping)
    

    def get_dataframe(self):
        return self.__df
    
    def get_x(self):
        X = self.__df.values
        if self.__x_preprocessor is not None:
            X = self.__x_preprocessor.transform(X)
        
        return X

    def form_test_file(self, prediction_function:Callable[[], np.ndarray]):
        def write_output():
            predictions = prediction_function()
            df_final = pd.DataFrame({"Id":self.ids, "SalePrice":predictions.reshape(-1)})
            df_final.to_csv("final.csv",index=False)
        return write_output






