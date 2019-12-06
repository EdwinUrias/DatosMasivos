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