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
//////////////////////////////////////////////////////DECISION TREE////////////////////////////////////////////////////////////////////////////
//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)
//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//DecisionTree
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
// features con mas de 4 valores distinctivos son tomados como continuos
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
//Division de los datos entre 70% y 30% en un arreglo
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
//Creamos un objeto DecisionTree
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Rama de prediccion
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Juntamos los datos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//Create a model of the entraining
val model = pipeline.fit(trainingData)
//Transformacion de datos en el modelo
val predictions = model.transform(testData)
//Desplegamos predicciones
predictions.select("predictedLabel", "label", "features").show(5)
//Evaluamos la exactitud
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
//println(s"Test Error = ${(1.0 - accuracy)}")
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
///////////////////////////////////////////////////////LOGISTIC REGRESION//////////////////////////////////////////////////////////////////////////////////////
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Logistic Regresion
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
//////////////////////////////////////////////////////MULTILAYER PERCEPTRON///////////////////////////////////////////////////////////////////////////////////
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Multilayer perceptron dividimos los datos en un arreglo en partes de 70% y 30%
val split = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1)
// Especificamos las capas para la red neuronal de entrada 5 por el numero de datos de las features 2 capas ocultas de dos neuronas la salida de 4  asi lo marca las clases
val layers = Array[Int](5, 2, 2, 4)
//Creamos el entrenador con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Entrenamos el modelo
val model = trainer.fit(train)
//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
/////////////////////////////////////////////////SUPORT VECTOR MACHINE///////////////////////////////////////////////////////////////////////////////
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//SVM
val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// Fit del modelo
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