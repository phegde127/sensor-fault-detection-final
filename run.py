from sensor.pipeline.training_pipeline_ph import TrainPipeline
from sensor.pipeline.prediction_pipeline_ph import PredictionPipeline

training_pipeline = TrainPipeline()
training_pipeline.run_pipeline()

prediction_pipeline = PredictionPipeline()
prediction_pipeline.initiate_prediction()

