from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from sensor.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelPusherConfig,ModelEvaluationConfig,ModelTrainerConfig
from sensor.exception import SensorException
import sys,os
from sensor.logger import logging
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

# training_pipeline_config = TrainingPipelineConfig()

# def start_data_ingestion()->DataIngestionArtifact:
#     try:
#         data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
#         logging.info("Starting data ingestion")
#         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
#         data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
#         logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
#         return data_ingestion_artifact
#     except  Exception as e:
#         raise  SensorException(e,sys)

# data_ingestion_artifact:DataIngestionArtifact = start_data_ingestion()

class TrainPipeline:
    #is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        #self.s3_sync = S3Sync()
        


    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def run_pipeline(self):
        try:
            
            #TrainPipeline.is_pipeline_running=True

            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            # data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            # data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            # model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            # if not model_eval_artifact.is_model_accepted:
            #     raise Exception("Trained model is not better than the best model")
            # model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            # TrainPipeline.is_pipeline_running=False
            #self.sync_artifact_dir_to_s3()
            #self.sync_saved_model_dir_to_s3()
        except  Exception as e:
            #self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise  SensorException(e,sys)