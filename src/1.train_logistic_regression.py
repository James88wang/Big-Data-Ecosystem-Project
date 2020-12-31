from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
import os
from os.path import isfile, join
import json

spark = SparkSession.builder.getOrCreate()

loc = os.path.abspath("")
data_loc = f"{loc}/data/creditcard.csv"


df = spark.read.csv(data_loc, inferSchema=True, header=True)

df = df.select('Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class')


train, test = df.randomSplit([0.8, 0.2], seed=7)


columns = [x for x in df.columns if x != 'Class']

vector_assembler = VectorAssembler(
    inputCols=columns, outputCol="features"
)

pipeline = Pipeline(stages=[vector_assembler])
modelLR = pipeline.fit(train)

predictionsLR = modelLR.transform(test)

predictionsLR.select(
    'features',
).show(truncate=False)


data = predictionsLR.select("features","Class")

data.show(5, truncate=False)

data = data.withColumnRenamed('Class','label')

modelLR = LogisticRegression().fit(data)

accuracy = modelLR.summary.accuracy
precision = modelLR.summary.weightedPrecision
recall = modelLR.summary.weightedRecall
f1 = modelLR.summary.weightedFMeasure()

if not os.path.exists(loc+'/scores'):
	os.mkdir(loc+'/scores')
with open(loc+'/scores/metrics.json', 'w') as fd:
	json.dump({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, fd)

modelLR.write().overwrite().save(loc+'/models/LR_model')














