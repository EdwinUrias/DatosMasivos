# DatosMasivos

## CODE
```scala
//Contenido del proyecto
//1.- Objectivo: Comparacion del rendimiento siguientes algoritmos de machine learning
// - SVM
// - Decision Three
// - Logistic Regresion
// - Multilayer perceptron
//Con el siguiente data set: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing 

// Contenido del documento de proyecto final
// 1. Portada
// 2. Indice
// 3. Introduccion
// 4. Marco teorico de los algoritmos
// 5. Implementacion (Que herramientas usaron y porque (en este caso spark con scala))
// 6. Resultados (Un tabular con los datos por cada algoritmo para ver su preformance)
//    y su respectiva explicacion.
// 7. Conclusiones
// 8. Referencias (No wikipedia por ningun motivo, traten que sean de articulos cientificos)
//    El documento debe estar referenciado 

// Nota: si el documento no es presentado , no revisare su desarrollo del proyecto
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
df.show(1)
//CAMBIAMOS COLUMNA POR UNA DE DATOS BINARIOS
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//MOSTRAMOS LA NUEVA COLUMNA
newcolumn.show(1)
//CREAMOS LA TABLA FEATURES
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//MOSTRAMOS LA TABLA FEATURES
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)

//////////////////////////////////////////////////////DECISION TREE////////////////////////////////////////////////////////////////////////////
//DECISION TREE
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
//FEATURES CON MAS DE 4 VALORES DISTINTIVOS TOMADOS COMO CONTINUOS
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
//70 PORCIENDO DE DATOS PARA ENTRETAR Y LOS OTROS 30 PORCIENTO PARA PRUEVAS
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
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
predictions.select("predictedLabel", "label", "features").show(5)
//EVALUACION DE LA EXACTITUD
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

///////////////////////////////////////////////////////LOGISTIC REGRESION//////////////////////////////////////////////////////////////////////////////////////
//Logistic Regresion
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)

//////////////////////////////////////////////////////MULTILAYER PERCEPTRON///////////////////////////////////////////////////////////////////////////////////
//MULTILAYER PERCEPTRON DIVIDIMOS LOS DATOS EN PARTES DE 60 Y 40 PORCIENTO RESPECTIVAMENTE
val split = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1) 
//ESPECIFICAMOS LAS CAPAS PARA LA RED NEURONAL DE 5 ENTRADAS POR EL NUMERO DE DATOS DE LOS FEATURES 2 CAPAS OCULTAS DE 2 NEURONAS
// Y LA SALIDA DE 4 POR QUE ASI LO MARCA LAS CLASES
val layers = Array[Int](5, 2, 2, 4)
//CREAMOS CONTENEDOR CON SOS PARAMETROS
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//ENTRENAMOS EL MODELO
val model = trainer.fit(train)
//IMPRIMIMOS LA EXACTITUD
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

/////////////////////////////////////////////////SUPORT VECTOR MACHINE///////////////////////////////////////////////////////////////////////////////
//SVM
val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// FIT DEL MODELO
val linsvcModel = linsvc.fit(c3)

//////////////////////////////////////////////RESULTADOS////////////////////////////////////////////////////////////////////////
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS DECISION TREE")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Test Error = ${(1.0 - accuracy)}")
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
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
//
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"SUPORT VECTOR MACHINE")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")
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
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@64a74723
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

+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown| no|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown|  2|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_04de40b5e58d
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|            features|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|management|married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown|  2|[2143.0,5.0,261.0...|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 1 row

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_0e9c6622932f
featureIndexer: org.apache.spark.ml.feature.VectorIndexer = vecIdx_abed9e5e23e1
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_01bd5865886e
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_d0d605e90c9d
pipeline: org.apache.spark.ml.Pipeline = pipeline_dbf9c9fe493c
model: org.apache.spark.ml.PipelineModel = pipeline_dbf9c9fe493c
predictions: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|             2|    1|[-1944.0,7.0,623....|
|             2|    1|[-1206.0,15.0,382...|
|             2|    1|[-970.0,4.0,489.0...|
|             2|    1|[-930.0,14.0,786....|
|             2|    1|[-770.0,18.0,618....|
+--------------+-----+--------------------+
only showing top 5 rows

evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_eaf3da357f65
accuracy: Double = 0.8878435984124651
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_01bd5865886e) of depth 5 with 33 nodes
logistic: org.apache.spark.ml.classification.LogisticRegression = logreg_b36cb90d6709
19/12/09 20:13:48 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
19/12/09 20:13:48 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
logisticModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_b36cb90d6709, numClasses = 3, numFeatures = 5
org.apache.spark.SparkException: Multinomial models contain a matrix of coefficients, use coefficientMatrix instead.
  at org.apache.spark.ml.classification.LogisticRegressionModel.coefficients(LogisticRegression.scala:955)
  ... 79 elided
logisticMult: org.apache.spark.ml.classification.LogisticRegression = logreg_a5ea03612448
logisticMultModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_a5ea03612448, numClasses = 3, numFeatures = 5
split: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, features: vector], [label: int, features: vector])
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
layers: Array[Int] = Array(5, 2, 2, 4)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_295643f36faf
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_295643f36faf
result: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 3 more fields]
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: int]
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_f2c696e7b419
c1: org.apache.spark.sql.DataFrame = [label: int, features: vector]
c2: org.apache.spark.sql.DataFrame = [label: int, features: vector]
c3: org.apache.spark.sql.DataFrame = [label: int, features: vector]
linsvc: org.apache.spark.ml.classification.LinearSVC = linearsvc_c7fbf63a469c
linsvcModel: org.apache.spark.ml.classification.LinearSVCModel = linearsvc_c7fbf63a469c


/////////////////////////////////////////////////////////////////////
RESULTADOS DECISION TREE
/////////////////////////////////////////////////////////////////////
Test Error = 0.11215640158753493
Learned classification tree model:
 DecisionTreeClassificationModel (uid=dtc_01bd5865886e) of depth 5 with 33 nodes
  If (feature 2 <= 559.5)
   If (feature 3 <= 9.5)
    Predict: 0.0
   Else (feature 3 > 9.5)
    If (feature 2 <= 174.5)
     Predict: 0.0
    Else (feature 2 > 174.5)
     If (feature 3 <= 187.5)
      If (feature 3 <= 92.5)
       Predict: 1.0
      Else (feature 3 > 92.5)
       Predict: 0.0
     Else (feature 3 > 187.5)
      If (feature 3 <= 519.5)
       Predict: 0.0
      Else (feature 3 > 519.5)
       Predict: 1.0
  Else (feature 2 > 559.5)
   If (feature 2 <= 879.0)
    If (feature 3 <= 0.0)
     If (feature 2 <= 669.5)
      Predict: 0.0
     Else (feature 2 > 669.5)
      If (feature 1 <= 29.5)
       Predict: 0.0
      Else (feature 1 > 29.5)
       Predict: 1.0
    Else (feature 3 > 0.0)
     If (feature 1 <= 20.5)
      If (feature 1 <= 16.5)
       Predict: 1.0
      Else (feature 1 > 16.5)
       Predict: 0.0
     Else (feature 1 > 20.5)
      If (feature 0 <= -82.5)
       Predict: 0.0
      Else (feature 0 > -82.5)
       Predict: 1.0
   Else (feature 2 > 879.0)
    If (feature 3 <= 272.5)
     Predict: 1.0
    Else (feature 3 > 272.5)
     If (feature 4 <= 3.5)
      If (feature 0 <= 71.5)
       Predict: 1.0
      Else (feature 0 > 71.5)
       Predict: 0.0
     Else (feature 4 > 3.5)
      Predict: 0.0



/////////////////////////////////////////////////////////////////////
RESULTADOS LOGISTIC REGRESION
/////////////////////////////////////////////////////////////////////
Multinomial coefficients: 3 x 5 CSCMatrix
Multinomial intercepts: [-7.827431229384973,2.903059293515478,4.924371935869495]


/////////////////////////////////////////////////////////////////////
RESULTADOS MULTILAYER PERCEPTRON
/////////////////////////////////////////////////////////////////////
Test set accuracy = 0.8848956335944776


/////////////////////////////////////////////////////////////////////
SUPORT VECTOR MACHINE
/////////////////////////////////////////////////////////////////////
Coefficients: [2.125897501491213E-6,0.013517727458849872,-7.514021888017163E-4,-2.7022337506408964E-4,-0.011177544540215354] Intercept: 1.084924165339881
```
