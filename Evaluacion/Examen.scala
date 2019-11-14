

//TODO ***************************************************************************************************************************************
//******Librerias
//import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline

//Se crea la estructura de los datos por columnas
val structtype = StructType(StructField("c0", DoubleType, true) ::StructField("c1", DoubleType, true) ::StructField("c2", DoubleType, true) ::StructField("c3",DoubleType, true) ::StructField("c4", StringType, true) :: Nil)
//Ahora cargamos el CSV como DataFrame con la estructura de columnas creada previamente
val dfstruct = spark.read.option("header", "false").schema(structtype)csv("/media/erebus/ESD-USB/Evaluacion/Iris.csv")
//Creamos la columna label
val label = new StringIndexer().setInputCol("c4").setOutputCol("label")
//Se crea un arreglo de los datos de las columnas c0-c3 en la columna features
val assembler = new VectorAssembler().setInputCols(Array("c0", "c1", "c2", "c3")).setOutputCol("features")

//Se separa los datos en dos grupos, uno para entrenar y el otro para prueba desde nuestro df
val splits = dfstruct.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0) //entrenar
val test = splits(1) //prueba

//Especificamos las capas de nuestra red neuronal
//4 de entrada, dos capas internas, una de 5, otra de 4 neuronas y 3 de salida
val layers = Array[Int](4, 7, 6, 3)

//Creamos el entrenador y especificamos los parametros
//.setLayers es para cargar las capas de nuestra red neuronal
//.setMaxIter es para indicar el numero maximo de iteraciones 
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)

//Se genera un pipeline con los datos que necesitamos de la label y las features 
val pipe = new Pipeline().setStages(Array(label,assembler,trainer))
//Ahora se entrena el modelo con los datos anteriores
val model = pipe.fit(train)

//Calculamos la exactitud en el conjunto test
val res = model.transform(test)
//Mostramos el resultado
res.show()
val predictionAndLabels = res.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Imprimimos los resultados de exactitud utilizando un evaluador multiclase
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")