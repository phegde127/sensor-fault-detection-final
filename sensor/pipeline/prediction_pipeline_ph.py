import os, sys
import pandas as pd
import numpy as np
from pandas import DataFrame

from sensor.cloud_storage.aws_storage import SimpleStorageService
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.config_entity import PredictionPipelineConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.ml.model.s3_estimator import SensorEstimator
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME, PREDICTION_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, PREPROCSSING_OBJECT_FILE_NAME, MODEL_FILE_NAME, TARGET_COLUMN

import awswrangler as wr
import pickle
from io import BytesIO 
from sklearn.impute import SimpleImputer

class PredictionPipeline:
    def __init__(
        self,
        prediction_pipeline_config: PredictionPipelineConfig = PredictionPipelineConfig(),
    ) -> None:
        """
        :param prediction_pipeline_config: 
        """
        try:
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
            self.s3 = SimpleStorageService()
            self.data_path = f"s3://{self.prediction_pipeline_config.data_bucket_name}/{self.prediction_pipeline_config.data_file_path}"
            self.transformation_object_path = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}/{PREPROCSSING_OBJECT_FILE_NAME}"
            self.model_object_path = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}/{MODEL_FILE_NAME}"
            

        except Exception as e:
            raise SensorException(e, sys)
    
    def get_data(self) -> DataFrame:
        try:
            logging.info("Entered get_data method of SensorData class")

            data_path = f"s3://{self.prediction_pipeline_config.data_bucket_name}/{self.prediction_pipeline_config.data_file_path}"
            
            prediction_df = wr.s3.read_csv(path=data_path)
        
            logging.info("Read prediction csv file from s3 bucket")

            prediction_df = prediction_df.drop(
                self.schema_config["drop_columns"], axis=1
            )


            logging.info("Dropped the required columns")

            logging.info("Exited the get_data method of SensorData class")

            return prediction_df

        except Exception as e:
            raise SensorException(e, sys)
    
    def get_object(self,obj_path):
        file_obj=obj_path

        try:
            with BytesIO() as f:
                wr.s3.download(path=file_obj, local_file=f)
                f.seek(0)
                model = pickle.load(f)

                return model

        except  Exception as e:
            raise  SensorException(e,sys)

       
    def predict(self) -> np.ndarray:
        try:
            logging.info("Entered predict method of SensorData class")

            data = self.get_data()
            input_data = data.drop(columns=[TARGET_COLUMN], axis=1)
            input_data.replace({"na": np.nan}, inplace=True)
            #simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            #input_arr =simple_imputer.fit_transform(input_data)

            output_data = data[TARGET_COLUMN]
            output_data = output_data.replace(TargetValueMapping().to_dict())

            print(input_data.head(5))
   
            model_pre = self.get_object(obj_path=self.transformation_object_path)
            data_arr = model_pre.transform(input_data)
            model = self.get_object(obj_path=self.model_object_path)
            predict = model.predict(data_arr)

            return predict

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_prediction(self,) -> None:
        try:

            dataframe = self.get_data()

            predicted_arr = self.predict()

            prediction = pd.DataFrame(list(predicted_arr))

            prediction.columns = ["class_prediction"]

            prediction.replace(TargetValueMapping().reverse_mapping(), inplace=True)

            predicted_dataframe = pd.concat([dataframe, prediction], axis=1)

            self.s3.upload_df_as_csv(
                predicted_dataframe,
                self.prediction_pipeline_config.output_file_name,
                self.prediction_pipeline_config.output_file_name,
                self.prediction_pipeline_config.data_bucket_name,
                )

            logging.info("Uploaded artifacts folder to s3 bucket_name")

            logging.info(f"File has uploaded to {predicted_dataframe}")

        except Exception as e:
            raise SensorException(e, sys)
            

#prediction_pipeline = PredictionPipeline()
#prediction_pipeline.initiate_prediction()

            #model_s3 = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}/{MODEL_FILE_NAME}"
            #s3_cls = SimpleStorageService()
            #model = s3_cls.load_model(MODEL_FILE_NAME, TRAINING_BUCKET_NAME)
            # data = wr.s3.read_csv(path="s3://ph-sensor-predict/test.csv")
            # y_true = data[TARGET_COLUMN]
            # y_true.replace(TargetValueMapping().to_dict(),inplace=True)
            # data.drop(TARGET_COLUMN,axis=1,inplace=True)

            # dir_path = s3DwonloadConfig(training_pipeline_config=self.training_pipeline_config)
            # print(dir_path.s3download_file_path)
            # os.makedirs(dir_path.s3download_file_path, exist_ok=True)
            # aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}/{MODEL_FILE_NAME}"
            # aws_buket_url_trans = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}/{DATA_TRANSFORMATION_DIR_NAME}/{DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR}/{PREPROCSSING_OBJECT_FILE_NAME}"
            # #print(aws_buket_url)
            # #print(aws_buket_url_trans)
            # #print(dir_path.s3download_prepro_file)
            # model = wr.s3.download(path=aws_buket_url, local_file=dir_path.s3download_model_file)
            # model_trans = wr.s3.download(path=aws_buket_url_trans, local_file=dir_path.s3download_prepro_file)
            # #print(model)
            # model_pre = pickle.load(open(dir_path.s3download_prepro_file, 'rb'))
            # data_arr = model_pre.transform(data)
            # model = pickle.load(open(dir_path.s3download_model_file, 'rb'))
            # predict = model.predict(data_arr)
            

            # print(predict)

            #return(predict)
            #print(data.head(5))
            #print(data_transformation_artifact.transformed_object_file_path)









