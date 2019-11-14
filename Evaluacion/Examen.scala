import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StringIndexer
//Columnas de datos
val structtype = StructType(StructField("c0", DoubleType, true) ::StructField("c1", DoubleType, true) ::StructField("c2", DoubleType, true) ::StructField("c3",DoubleType, true) ::StructField("c4", StringType, true) :: Nil)
//Carga del dataset
val dfstruct = spark.read.option("header", "false").schema(structtype)csv("/home/erebus/DatosMasivos/Evaluacion/Iris.csv")
//Columna etiqueta label
val label = new StringIndexer().setInputCol("c4").setOutputCol("label")
//Arreglo de datos de columnas c0 c1 c2 c3 en la columna features
val assembler = new VectorAssembler().setInputCols(Array("c0", "c1", "c2", "c3")).setOutputCol("features")
//Separa los datos en dos grupos (split)para entrenar y ralizar prueba desde df
val splits = dfstruct.randomSplit(Array(0.7, 0.3), seed = 1234L)
//Val de entrenamiento
val train = splits(0) 
//Val de pruebas
val test = splits(1)
//Especificamos las capas de nuestra red neuronal 4 neuronas de entrada, dos capas internasy 3 de salida
val layers = Array[Int](4, 5, 3, 3)
//Se crea el tester y se especifican los parametros. La propiedad "setLayers" carga las capas de la red neuronal y la pripiedad "setMaxIter" indica el numero maximo de iteraciones 
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Genera un pipeline con los datos para el label y los features 
val pipe = new Pipeline().setStages(Array(label,assembler,trainer))
//Con los datos entrena el modelo
val model = pipe.fit(train)
//Se calcula la presicion del test
val res = model.transform(test)
//Muestra el resultado
res.show()
val predictionAndLabels = res.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Imprime resultados de presicion utilizando un evaluador multiclase
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")