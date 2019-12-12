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
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// FIT DEL MODELO
val linsvcModel = linsvc.fit(cambio3)
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"SUPORT VECTOR MACHINE")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")
//////////////////////////////////////////////////////DECISION TREE////////////////////////////////////////////////////////////////////////////
//DECISION TREE
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
//FEATURES CON MAS DE 4 VALORES DISTINTIVOS TOMADOS COMO CONTINUOS
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
//80% DE DATOS PARA ENTRETAR Y LOS OTROS 20% PARA PRUEVAS
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
predictions.select("predictedLabel", "label", "features").show(10)
//EVALUACION DE LA EXACTITUD
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS DECISION TREE")
println(s"/////////////////////////////////////////////////////////////////////")
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

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
println(s"")
println(s"")
println(s"/////////////////////////////////////////////////////////////////////")
println(s"RESULTADOS MULTILAYER PERCEPTRON")
println(s"/////////////////////////////////////////////////////////////////////")
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")