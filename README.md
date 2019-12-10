# DatosMasivos

## CODE
```scala
//////////////////////////////////////////////////////SETUP//////////////////////////////////////////////////////////////////
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
//REDUCIMOS LOS ERRORES O ADVERTENCIAS
Logger.getLogger("org").setLevel(Level.ERROR)
//CREAMOS SESION EN SPARK Y CARGAMOS EL CSV
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//IMPRIMIMOS LOS TIPOS DE DATOS
df.printSchema()
df.show(10)
//CAMBIAMOS COLUMNA POR UNA DE DATOS BINARIOS
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//MOSTRAMOS LA NUEVA COLUMNA
newcolumn.show(10)
//CREAMOS LA TABLA FEATURES
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val ft = assembler.transform(newcolumn)
//MOSTRAMOS LA TABLA FEATURES
ft.show(10)
//SE CAMBIA LA COLUMNA Y LA COLUMNA LABEL
val cambio = ft.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(10)
/////////////////////////////////////////////////SUPORT VECTOR MACHINE///////////////////////////////////////////////////////////////////////////////
//SVM
val cambio1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val cambio2 = cambio1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val cambio3 = cambio2.withColumn("label",'label.cast("Int"))
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.2)
// FIT DEL MODELO
val linsvcModel = linsvc.fit(cambio3)
//////////////////////////////////////////////////////DECISION TREE////////////////////////////////////////////////////////////////////////////
//DECISION TREE
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
//FEATURES CON MAS DE 4 VALORES DISTINTIVOS TOMADOS COMO CONTINUOS
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
//80% DE DATOS PARA ENTRETAR Y LOS OTROS 20% PARA PRUEVAS
val Array(trainingData, testData) = feat.randomSplit(Array(0.8, 0.2))
//CREA EL OBJETO DECISION TREE
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//RAMA PARA PREDICCION
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//UNION DE DATOS DEN PIPELINE
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//CREACION DE UN MODELO PARA ENTRENAR
val model = pipeline.fit(trainingData)
//TTRANSFORMACION DEL MODELO PARA LOS DATOS DE ENTRENAMIENTO
val predictions = model.transform(testData)
//DESPLEGAMOS LAS PREDICCIONES
predictions.select("predictedLabel", "label", "features").show(10)
//EVALUACION DE LA EXACTITUD
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
///////////////////////////////////////////////////////LOGISTIC REGRESION//////////////////////////////////////////////////////////////////////////////////////
//Logistic Regresion
val logistic = new LogisticRegression().setMaxIter(5).setRegParam(0.2).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticReg
ression().setMaxIter(5).setRegParam(0.2).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
//////////////////////////////////////////////////////MULTILAYER PERCEPTRON///////////////////////////////////////////////////////////////////////////////////
//MULTILAYER PERCEPTRON DIVIDIMOS LOS DATOS EN PARTES DE 80 Y 20 PORCIENTO RESPECTIVAMENTE
val split = feat.randomSplit(Array(0.8, 0.2), seed = 1234L)
val train = split(0)
val test = split(1) 
//ESPECIFICAMOS LAS CAPAS PARA LA RED NEURONAL DE 5 ENTRADAS POR EL NUMERO DE DATOS DE LOS FEATURES 4 CAPAS OCULTAS DE 5 NEURONAS
// Y LA SALIDA DE 4 POR QUE ASI LO MARCA LAS CLASES
val layers = Array[Int](5, 4, 5, 4)
//CREAMOS CONTENEDOR CON SOS PARAMETROS
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//ENTRENAMOS EL MODELO
val model = trainer.fit(train)
//IMPRIMIMOS LA EXACTITUD
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//////////////////////////////////////////////RESULTADOS/////////////////////////////////////////////////////////////////////////
//
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"SUPORT VECTOR MACHINE")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")
//
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS DECISION TREE")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Test Error = ${(1.0 - accuracy)}")
val accuracy = evaluator.evaluate(predictions)
//
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS LOGISTIC REGRESION")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
//
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS MULTILAYER PERCEPTRON")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
## VALOR

```sh
scala> :load Proyecto.scala
Loading Proyecto.scala...
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@18a9e8db
df: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
root
 |-- age: integer (nullable = true)
 |-- job: string (nullable = true)
 |-- marital: string (nullable = true)
 |-- education: string (nullable = true)
 |-- default: string (nullable = true)
 |-- balance: integer (nullable = true)
 |-- housing: string (nullable = true)
 |-- loan: string (nullable = true)
 |-- contact: string (nullable = true)
 |-- day: integer (nullable = true)
 |-- month: string (nullable = true)
 |-- duration: integer (nullable = true)
 |-- campaign: integer (nullable = true)
 |-- pdays: integer (nullable = true)
 |-- previous: integer (nullable = true)
 |-- poutcome: string (nullable = true)
 |-- y: string (nullable = true)

+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|         job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|  management| married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown| no|
| 44|  technician|  single|secondary|     no|     29|    yes|  no|unknown|  5|  may|     151|       1|   -1|       0| unknown| no|
| 33|entrepreneur| married|secondary|     no|      2|    yes| yes|unknown|  5|  may|      76|       1|   -1|       0| unknown| no|
| 47| blue-collar| married|  unknown|     no|   1506|    yes|  no|unknown|  5|  may|      92|       1|   -1|       0| unknown| no|
| 33|     unknown|  single|  unknown|     no|      1|     no|  no|unknown|  5|  may|     198|       1|   -1|       0| unknown| no|
| 35|  management| married| tertiary|     no|    231|    yes|  no|unknown|  5|  may|     139|       1|   -1|       0| unknown| no|
| 28|  management|  single| tertiary|     no|    447|    yes| yes|unknown|  5|  may|     217|       1|   -1|       0| unknown| no|
| 42|entrepreneur|divorced| tertiary|    yes|      2|    yes|  no|unknown|  5|  may|     380|       1|   -1|       0| unknown| no|
| 58|     retired| married|  primary|     no|    121|    yes|  no|unknown|  5|  may|      50|       1|   -1|       0| unknown| no|
| 43|  technician|  single|secondary|     no|    593|    yes|  no|unknown|  5|  may|      55|       1|   -1|       0| unknown| no|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 10 rows

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|         job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|  management| married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown|  2|
| 44|  technician|  single|secondary|     no|     29|    yes|  no|unknown|  5|  may|     151|       1|   -1|       0| unknown|  2|
| 33|entrepreneur| married|secondary|     no|      2|    yes| yes|unknown|  5|  may|      76|       1|   -1|       0| unknown|  2|
| 47| blue-collar| married|  unknown|     no|   1506|    yes|  no|unknown|  5|  may|      92|       1|   -1|       0| unknown|  2|
| 33|     unknown|  single|  unknown|     no|      1|     no|  no|unknown|  5|  may|     198|       1|   -1|       0| unknown|  2|
| 35|  management| married| tertiary|     no|    231|    yes|  no|unknown|  5|  may|     139|       1|   -1|       0| unknown|  2|
| 28|  management|  single| tertiary|     no|    447|    yes| yes|unknown|  5|  may|     217|       1|   -1|       0| unknown|  2|
| 42|entrepreneur|divorced| tertiary|    yes|      2|    yes|  no|unknown|  5|  may|     380|       1|   -1|       0| unknown|  2|
| 58|     retired| married|  primary|     no|    121|    yes|  no|unknown|  5|  may|      50|       1|   -1|       0| unknown|  2|
| 43|  technician|  single|secondary|     no|    593|    yes|  no|unknown|  5|  may|      55|       1|   -1|       0| unknown|  2|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 10 rows

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_9f580bee940d
ft: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|         job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|            features|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|  management| married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown|  2|[2143.0,5.0,261.0...|
| 44|  technician|  single|secondary|     no|     29|    yes|  no|unknown|  5|  may|     151|       1|   -1|       0| unknown|  2|[29.0,5.0,151.0,-...|
| 33|entrepreneur| married|secondary|     no|      2|    yes| yes|unknown|  5|  may|      76|       1|   -1|       0| unknown|  2|[2.0,5.0,76.0,-1....|
| 47| blue-collar| married|  unknown|     no|   1506|    yes|  no|unknown|  5|  may|      92|       1|   -1|       0| unknown|  2|[1506.0,5.0,92.0,...|
| 33|     unknown|  single|  unknown|     no|      1|     no|  no|unknown|  5|  may|     198|       1|   -1|       0| unknown|  2|[1.0,5.0,198.0,-1...|
| 35|  management| married| tertiary|     no|    231|    yes|  no|unknown|  5|  may|     139|       1|   -1|       0| unknown|  2|[231.0,5.0,139.0,...|
| 28|  management|  single| tertiary|     no|    447|    yes| yes|unknown|  5|  may|     217|       1|   -1|       0| unknown|  2|[447.0,5.0,217.0,...|
| 42|entrepreneur|divorced| tertiary|    yes|      2|    yes|  no|unknown|  5|  may|     380|       1|   -1|       0| unknown|  2|[2.0,5.0,380.0,-1...|
| 58|     retired| married|  primary|     no|    121|    yes|  no|unknown|  5|  may|      50|       1|   -1|       0| unknown|  2|[121.0,5.0,50.0,-...|
| 43|  technician|  single|secondary|     no|    593|    yes|  no|unknown|  5|  may|      55|       1|   -1|       0| unknown|  2|[593.0,5.0,55.0,-...|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 10 rows

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
|    2|[29.0,5.0,151.0,-...|
|    2|[2.0,5.0,76.0,-1....|
|    2|[1506.0,5.0,92.0,...|
|    2|[1.0,5.0,198.0,-1...|
|    2|[231.0,5.0,139.0,...|
|    2|[447.0,5.0,217.0,...|
|    2|[2.0,5.0,380.0,-1...|
|    2|[121.0,5.0,50.0,-...|
|    2|[593.0,5.0,55.0,-...|
+-----+--------------------+
only showing top 10 rows

cambio1: org.apache.spark.sql.DataFrame = [label: int, features: vector]
cambio2: org.apache.spark.sql.DataFrame = [label: int, features: vector]
cambio3: org.apache.spark.sql.DataFrame = [label: int, features: vector]
linsvc: org.apache.spark.ml.classification.LinearSVC = linearsvc_f8a705be6598
linsvcModel: org.apache.spark.ml.classification.LinearSVCModel = linearsvc_f8a705be6598
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_5c691827940a
featureIndexer: org.apache.spark.ml.feature.VectorIndexer = vecIdx_8d3b23bf5b8e
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_08e35ad4bfa0
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_5404bdb77934
pipeline: org.apache.spark.ml.Pipeline = pipeline_7b361a9d62e7
model: org.apache.spark.ml.PipelineModel = pipeline_7b361a9d62e7
predictions: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|             2|    1|[-1206.0,15.0,382...|
|             2|    1|[-770.0,18.0,618....|
|             1|    1|[-725.0,27.0,1205...|
|             2|    1|[-639.0,15.0,585....|
|             1|    1|[-546.0,25.0,1152...|
|             1|    1|[-454.0,18.0,801....|
|             2|    1|[-449.0,14.0,691....|
|             2|    1|[-416.0,16.0,767....|
|             2|    1|[-413.0,27.0,422....|
|             2|    1|[-407.0,12.0,829....|
+--------------+-----+--------------------+
only showing top 10 rows

evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_f7d5f75f148a
accuracy: Double = 0.8948476454293629
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_08e35ad4bfa0) of depth 5 with 25 nodes
Learned classification tree model:
 DecisionTreeClassificationModel (uid=dtc_08e35ad4bfa0) of depth 5 with 25 nodes
  If (feature 2 <= 478.5)
   If (feature 3 <= 9.5)
    Predict: 0.0
   Else (feature 3 > 9.5)
    If (feature 2 <= 158.5)
     Predict: 0.0
    Else (feature 2 > 158.5)
     If (feature 3 <= 191.5)
      If (feature 3 <= 95.5)
       Predict: 1.0
      Else (feature 3 > 95.5)
       Predict: 0.0
     Else (feature 3 > 191.5)
      Predict: 0.0
  Else (feature 2 > 478.5)
   If (feature 2 <= 659.5)
    If (feature 3 <= 8.5)
     Predict: 0.0
    Else (feature 3 > 8.5)
     If (feature 3 <= 191.5)
      Predict: 1.0
     Else (feature 3 > 191.5)
      If (feature 4 <= 12.5)
       Predict: 0.0
      Else (feature 4 > 12.5)
       Predict: 1.0
   Else (feature 2 > 659.5)
    If (feature 2 <= 856.0)
     If (feature 3 <= 0.0)
      If (feature 1 <= 29.5)
       Predict: 0.0
      Else (feature 1 > 29.5)
       Predict: 1.0
     Else (feature 3 > 0.0)
      Predict: 1.0
    Else (feature 2 > 856.0)
     Predict: 1.0

logistic: org.apache.spark.ml.classification.LogisticRegression = logreg_9809f7eca1dc
logisticModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_9809f7eca1dc, numClasses = 3, numFeatures = 5
org.apache.spark.SparkException: Multinomial models contain a matrix of coefficients, use coefficientMatrix instead.
  at org.apache.spark.ml.classification.LogisticRegressionModel.coefficients(LogisticRegression.scala:955)
  ... 94 elided
Proyecto.scala:167: error: not found: type LogisticReg
       val logisticMult = new LogisticReg
                              ^
Proyecto.scala:168: error: not found: value ression
       ression().setMaxIter(5).setRegParam(0.2).setElasticNetParam(0.8).setFamily("multinomial")
       ^
logisticMultModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_05d7fbbe8ab8, numClasses = 3, numFeatures = 5
split: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, features: vector], [label: int, features: vector])
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
layers: Array[Int] = Array(5, 4, 5, 4)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_a32c3f802b2f
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_a32c3f802b2f
result: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 3 more fields]
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: int]
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_7b04bd28d7ec


/////////////////////////////////////////////////////////////////////
SUPORT VECTOR MACHINE
/////////////////////////////////////////////////////////////////////
Coefficients: [-0.0,0.008569399668559616,-6.313630373326434E-4,-1.4295640198223574E-4,-0.012918141151966479] Intercept: 1.093914190592112


/////////////////////////////////////////////////////////////////////
RESULTADOS DECISION TREE
/////////////////////////////////////////////////////////////////////
Test Error = 0.1051523545706371
accuracy: Double = 0.036454293628808865


/////////////////////////////////////////////////////////////////////
RESULTADOS LOGISTIC REGRESION
/////////////////////////////////////////////////////////////////////
Multinomial coefficients: 3 x 5 CSCMatrix
Multinomial intercepts: [-6.832552186677845,2.4056393781318075,4.426912808546038]


/////////////////////////////////////////////////////////////////////
RESULTADOS MULTILAYER PERCEPTRON
/////////////////////////////////////////////////////////////////////
Test set accuracy = 0.8874070835155226
```
