# %load_ext nb_black
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
from os.path import isfile, join

spark = SparkSession.builder.getOrCreate()



loc = os.path.abspath("")
data_loc = f"{loc}/data/creditcard.csv"


df = spark.read.csv(data_loc, inferSchema=True, header=True)

df = df.select('Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class')


train, test = df.randomSplit([0.8, 0.2], seed=7)

from pyspark.ml.feature import VectorAssembler

columns = [x for x in df.columns if x != 'Class']

vector_assembler = VectorAssembler(
    inputCols=columns, outputCol="features"
)

RF = RandomForestClassifier(labelCol="Class", featuresCol="features", numTrees=10)

pipelineRF = Pipeline(stages=[vector_assembler, RF])

modelRF = pipelineRF.fit(train)

predictionsRF = modelRF.transform(test)

predictionsRF.select("prediction", "Class", "features").show(5)

evaluatorRF = MulticlassClassificationEvaluator(
    labelCol="Class", predictionCol="prediction", metricName="accuracy")
accuracyRF = evaluatorRF.evaluate(predictionsRF)


evaluatorRF.save('metrics/metricRF')

modelRF.write().overwrite().save('models/RF_model')



