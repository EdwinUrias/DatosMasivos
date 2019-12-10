# DatosMasivos

## CODE
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
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
//Especificamos las capas de nuestra red neuronal 4 neuronas de entrada, dos capas internas ocultas y 3 de salida
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
```
## VALOR
```sh
scala> :load Examen.scala
Loading Examen.scala...
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
structtype: org.apache.spark.sql.types.StructType = StructType(StructField(c0,DoubleType,true), StructField(c1,DoubleType,true), StructField(c2,DoubleType,true), StructField(c3,DoubleType,true), StructField(c4,StringType,true))
dfstruct: org.apache.spark.sql.DataFrame = [c0: double, c1: double ... 3 more fields]
label: org.apache.spark.ml.feature.StringIndexer = strIdx_83f35a5f6a4c
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_17f0e60ef42f
splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([c0: double, c1: double ... 3 more fields], [c0: double, c1: double ... 3 more fields])
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [c0: double, c1: double ... 3 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [c0: double, c1: double ... 3 more fields]
layers: Array[Int] = Array(4, 5, 3, 3)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_1ddd7d1a5876
pipe: org.apache.spark.ml.Pipeline = pipeline_cf267fb644b3
19/12/09 20:10:43 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
19/12/09 20:10:43 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
model: org.apache.spark.ml.PipelineModel = pipeline_cf267fb644b3
res: org.apache.spark.sql.DataFrame = [c0: double, c1: double ... 8 more fields]
+---+---+---+---+---------------+-----+-----------------+--------------------+--------------------+----------+
| c0| c1| c2| c3|             c4|label|         features|       rawPrediction|         probability|prediction|
+---+---+---+---+---------------+-----+-----------------+--------------------+--------------------+----------+
|4.3|3.0|1.1|0.1|    Iris-setosa|  1.0|[4.3,3.0,1.1,0.1]|[5.36861863240846...|[5.21836344819958...|       1.0|
|4.4|2.9|1.4|0.2|    Iris-setosa|  1.0|[4.4,2.9,1.4,0.2]|[5.36861863240858...|[5.21836344820306...|       1.0|
|4.4|3.0|1.3|0.2|    Iris-setosa|  1.0|[4.4,3.0,1.3,0.2]|[5.36861863240858...|[5.21836344820288...|       1.0|
|4.8|3.1|1.6|0.2|    Iris-setosa|  1.0|[4.8,3.1,1.6,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.0|3.3|1.4|0.2|    Iris-setosa|  1.0|[5.0,3.3,1.4,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.0|3.4|1.5|0.2|    Iris-setosa|  1.0|[5.0,3.4,1.5,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.0|3.6|1.4|0.2|    Iris-setosa|  1.0|[5.0,3.6,1.4,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.1|3.4|1.5|0.2|    Iris-setosa|  1.0|[5.1,3.4,1.5,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.1|3.8|1.5|0.3|    Iris-setosa|  1.0|[5.1,3.8,1.5,0.3]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.2|2.7|3.9|1.4|Iris-versicolor|  0.0|[5.2,2.7,3.9,1.4]|[22.2627257728457...|[0.99999999999998...|       0.0|
|5.2|4.1|1.5|0.1|    Iris-setosa|  1.0|[5.2,4.1,1.5,0.1]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.3|3.7|1.5|0.2|    Iris-setosa|  1.0|[5.3,3.7,1.5,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.4|3.4|1.5|0.4|    Iris-setosa|  1.0|[5.4,3.4,1.5,0.4]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.5|2.3|4.0|1.3|Iris-versicolor|  0.0|[5.5,2.3,4.0,1.3]|[22.2627257728457...|[0.99999999999998...|       0.0|
|5.6|2.9|3.6|1.3|Iris-versicolor|  0.0|[5.6,2.9,3.6,1.3]|[22.2627257728457...|[0.99999999999998...|       0.0|
|5.7|2.5|5.0|2.0| Iris-virginica|  2.0|[5.7,2.5,5.0,2.0]|[23.3392438330822...|[0.02777776040662...|       2.0|
|5.8|2.7|3.9|1.2|Iris-versicolor|  0.0|[5.8,2.7,3.9,1.2]|[22.2627257728457...|[0.99999999999998...|       0.0|
|5.8|2.8|5.1|2.4| Iris-virginica|  2.0|[5.8,2.8,5.1,2.4]|[23.3392438330822...|[0.02777776040662...|       2.0|
|5.8|4.0|1.2|0.2|    Iris-setosa|  1.0|[5.8,4.0,1.2,0.2]|[5.36861863240859...|[5.21836344820314...|       1.0|
|5.9|3.0|5.1|1.8| Iris-virginica|  2.0|[5.9,3.0,5.1,1.8]|[23.3392438330822...|[0.02777776040662...|       2.0|
+---+---+---+---+---------------+-----+-----------------+--------------------+--------------------+----------+
only showing top 20 rows

predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: double]
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_303f74a0545e
Test set accuracy = 1.0
```
