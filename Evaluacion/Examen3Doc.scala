zimport org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.sql.types._ 
import org.apache.spark.ml.Pipeline 
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

val estructuraIris = StructType(StructField("Datos0", DoubleType, true) ::StructField("Datos1", DoubleType, true) ::StructField("Datos2", DoubleType, true) ::StructField("Datos3",DoubleType, true) ::StructField("Datos4", StringType, true) :: Nil)

//Se carga Iris
val datosIris = spark.read.option("header", "true").schema(estructuraIris)csv("Iris.csv")
//val datosIris = spark.read.option("header", "false").format("csv").load("Iris.csv")
//datosIris.show()  

val etiqueta = new StringIndexer().setInputCol("Datos4").setOutputCol("label")
val ensamblador = new VectorAssembler().setInputCols(Array("Datos0", "Datos1", "Datos2", "Datos3")).setOutputCol("features")

//Datos de entrenamiento:70 y prueba:30
val splits = datosIris.randomSplit(Array(0.85, 0.15), seed = 1234L)
val entrenar = splits(0)
val prueba = splits(1)

//Capa entrada:4 neuronas. Capa intermedia:4,5 neuronas. Capa salida:3 neuronas
//Son 4 y 3 salidas: setosa, versicolor, virginica
val capasN = Array[Int](4, 8, 7, 3)

//Se hace la creacion del entrenador y se le dan los parametros
val entrenador = new MultilayerPerceptronClassifier().setLayers(capasN).setBlockSize(128).setSeed(1234L).setMaxIter(100)

val pipeline = new Pipeline().setStages(Array(etiqueta,ensamblador,entrenador))

//Entrenando el modelo 
val modelo = pipeline.fit(entrenar)
//val modelo = entrenado.fit(train)

val resultados = modelo.transform(prueba)
resultados.show()
val predictionAndLabels = resultados.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")