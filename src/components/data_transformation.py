import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_Object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            logging.info('initiating Scaling at transformation pipeline')

            numerical_columns=[ 'reading score', 'writing score'] 
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                  steps=[
                      ('imputer',SimpleImputer(strategy='most_frequent')),
                      ('one_hot_encoder',OneHotEncoder()),
                      ('scaler',StandardScaler(with_mean=False))


                  ]
            )
            logging.info('Scaling at transformation pipeline initiated')

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
           
            logging.info('Scaling at transformation pipeline completed')

            return preprocessor

        
        except Exception as e:
            raise CustomException(e,sys)
    
    logging.info('entering initiate_data_transformation')

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info(' Reading train and test data')

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(' Reading train and test data completed')

            logging.info('obtaining preprocessor object')
            
            target_column_name="math score"

            numerical_columns=[ 'reading score', 'writing score'] 
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info('applying preprocessor on train and test data')

            preprocessor_obj=self.get_data_transformer_obj()
            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_array,np.array(target_feature_test_df)


            ]
            logging.info('data processed')

            save_Object(
               file_path=self.data_config.preprocessor_obj_path,
               obj=preprocessor_obj
            )
            logging.info('saved preprocessoed object')


            return(
                train_arr,
                test_arr,
                self.data_config.preprocessor_obj_path



            )





        except Exception as e:
            raise CustomException(e,sys)