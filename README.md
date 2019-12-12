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
=======
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
```
## VALOR

```sh
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
=======
## CODE EVALUATION 1
```scala
var input0 = List(10,5,20,20,4,5,2,25,1)
var input1 = List(3,4,21,36,10,28,35,5,24,42)
//Define el ciclo
def breakingRecords (nums:List[Int]) : Unit =
{    
    //busca el mejor record
    var max, min = nums(0)
    //buscar el peor record
    var lowest, highest = 0
    //Inicia el recorrido
    for (i <- nums)
    {
        //Busca el lugar donde el siguiente lugar es el mejor record
        if (i>max)
        {max = i
            highest = highest + 1
            }
        //Busca el lugar donde el siguiente lugar es el peor record
        if (i<min)
        {min = i
            lowest = lowest +1
            }
    }
    //imprime el mejor record descues el peor record
    println (highest,lowest)
}
//Imprimir el input0
breakingRecords(input0)
//Imprimir el input1
breakingRecords(input1)
```

## VALOR
```sh
scala> :load examen1.scala
Loading examen1.scala...
input0: List[Int] = List(10, 5, 20, 20, 4, 5, 2, 25, 1)
input1: List[Int] = List(3, 4, 21, 36, 10, 28, 35, 5, 24, 42)
breakingRecords: (nums: List[Int])Unit
(2,4)
(4,0)
```

## CODE EVALUATION 2
```scala
import org.apache.spark.sql.SparkSession
import spark.implicits._
//1. Comienza una simple sesión Spark.
val spark = SparkSession.builder().getOrCreate()
//2. Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos.
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//3. ¿Cuáles son los nombres de las columnas?
df.columns
//4. ¿Cómo es el esquema?
df.printSchema()
//5. Imprime las primeras 5 columnas.
df.head(5)
df.show(5)
//6. Usa describe () para aprender sobre el DataFrame.
df.describe()
//7. Crea un nuevo dataframe con una columna nueva llamada “HV Ratio” que es la relación entre el precio de la columna 
//   “High” frente a la columna “Volumen” de acciones negociadas por un día.
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.select("HV Ratio").show()
//8. ¿Qué día tuvo el pico mas alto en la columna “Price”?
//--------------Pregunta cancelada!-----------------------
//9. ¿Cuál es el significado de la columna Cerrar “Close”?
print("Representa el precio con el que cerro el valor por accion de Netflix de ese dia.")
//10. ¿Cuál es el máximo y mínimo de la columna “Volumen”?
df.select(max("Volume")).show()
df.select(min("Volume")).show()
//11. Con Sintaxis Scala/Spark $ conteste los siguiente:
//a. ¿Cuántos días fue la columna “Close” inferior a $ 600?
df.filter($"Close" < 600).count()
//b. ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
print((df.filter($"High">500).count()*100)/1250,"%")
//c. ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
df.select(corr("High", "Volume")).show()
//d. ¿Cuál es el máximo de la columna “High” por año?
val df2=df.withColumn("Year",year(df("Date")))
val dfmax=df2.groupBy("Year").max()
dfmax.select($"Year",$"max(High)").show()
dfmax.select($"Year",$"max(High)").show(1)
//e. ¿Cuál es el promedio de columna “Close” para cada mes del calendario?
val dfmonth=df.withColumn("Month",month(df("Date")))
val dfmean=dfmonth.select($"Month",$"Close").groupBy("Month").mean()
dfmean.orderBy($"Month".desc).show()
dfmean.orderBy($"Month").show()
```
## VALOR
```sh
scala> :load Evaluacion1-2.scala
Loading Evaluacion1-2.scala...
import org.apache.spark.sql.SparkSession
import spark.implicits._
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@3a8dd899
df: org.apache.spark.sql.DataFrame = [Date: timestamp, Open: double ... 5 more fields]
res2: Array[String] = Array(Date, Open, High, Low, Close, Volume, Adj Close)
root
 |-- Date: timestamp (nullable = true)
 |-- Open: double (nullable = true)
 |-- High: double (nullable = true)
 |-- Low: double (nullable = true)
 |-- Close: double (nullable = true)
 |-- Volume: integer (nullable = true)
 |-- Adj Close: double (nullable = true)

res4: Array[org.apache.spark.sql.Row] = Array([2011-10-24 00:00:00.0,119.100002,120.28000300000001,115.100004,118.839996,120460200,16.977142], [2011-10-25 00:00:00.0,74.899999,79.390001,74.249997,77.370002,315541800,11.052857000000001], [2011-10-26 00:00:00.0,78.73,81.420001,75.399997,79.400002,148733900,11.342857], [2011-10-27 00:00:00.0,82.179998,82.71999699999999,79.249998,80.86000200000001,71190000,11.551428999999999], [2011-10-28 00:00:00.0,80.280002,84.660002,79.599999,84.14000300000001,57769600,12.02])
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
|               Date|      Open|              High|       Low|            Close|   Volume|         Adj Close|
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
|2011-10-24 00:00:00|119.100002|120.28000300000001|115.100004|       118.839996|120460200|         16.977142|
|2011-10-25 00:00:00| 74.899999|         79.390001| 74.249997|        77.370002|315541800|11.052857000000001|
|2011-10-26 00:00:00|     78.73|         81.420001| 75.399997|        79.400002|148733900|         11.342857|
|2011-10-27 00:00:00| 82.179998| 82.71999699999999| 79.249998|80.86000200000001| 71190000|11.551428999999999|
|2011-10-28 00:00:00| 80.280002|         84.660002| 79.599999|84.14000300000001| 57769600|             12.02|
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
only showing top 5 rows

res6: org.apache.spark.sql.DataFrame = [summary: string, Open: string ... 5 more fields]
df2: org.apache.spark.sql.DataFrame = [Date: timestamp, Open: double ... 6 more fields]
+--------------------+
|            HV Ratio|
+--------------------+
|9.985040951285156E-7|
|2.515989989281927E-7|
|5.474206014903126E-7|
|1.161960907430818...|
|1.465476686700271...|
|2.120614572195210...|
|2.453341026526372E-6|
|2.039435578967717E-6|
| 9.77974483949496E-7|
|1.099502069629999...|
|1.976194645910725...|
|2.902275528113834...|
|3.145082800111281E-6|
|2.279474054889131E-6|
|2.305965805108520...|
|4.039190694731629...|
|4.073010190713256...|
|2.501707242971725E-6|
|1.533411291208063...|
|2.274749388841058...|
+--------------------+
only showing top 20 rows

Representa el precio con el que cerro el valor por accion de Netflix de ese dia.+-----------+
|max(Volume)|
+-----------+
|  315541800|
+-----------+

+-----------+
|min(Volume)|
+-----------+
|    3531300|
+-----------+

res11: Long = 1218
(4,%)+--------------------+
|  corr(High, Volume)|
+--------------------+
|-0.20960233287942157|
+--------------------+

df2: org.apache.spark.sql.DataFrame = [Date: timestamp, Open: double ... 6 more fields]
dfmax: org.apache.spark.sql.DataFrame = [Year: int, max(Open): double ... 6 more fields]
+----+------------------+
|Year|         max(High)|
+----+------------------+
|2015|        716.159996|
|2013|        389.159988|
|2014|        489.290024|
|2012|        133.429996|
|2016|129.28999299999998|
|2011|120.28000300000001|
+----+------------------+

+----+----------+
|Year| max(High)|
+----+----------+
|2015|716.159996|
+----+----------+
only showing top 1 row

dfmonth: org.apache.spark.sql.DataFrame = [Date: timestamp, Open: double ... 6 more fields]
dfmean: org.apache.spark.sql.DataFrame = [Month: int, avg(Month): double ... 1 more field]
+-----+----------+------------------+
|Month|avg(Month)|        avg(Close)|
+-----+----------+------------------+
|   12|      12.0| 199.3700942358491|
|   11|      11.0| 194.3172275445545|
|   10|      10.0|205.93297300900903|
|    9|       9.0|206.09598121568627|
|    8|       8.0|195.25599892727263|
|    7|       7.0|243.64747528037387|
|    6|       6.0| 295.1597153490566|
|    5|       5.0|264.37037614150944|
|    4|       4.0|246.97514271428562|
|    3|       3.0| 249.5825228971963|
|    2|       2.0| 254.1954634020619|
|    1|       1.0|212.22613874257422|
+-----+----------+------------------+

+-----+----------+------------------+
|Month|avg(Month)|        avg(Close)|
+-----+----------+------------------+
|    1|       1.0|212.22613874257422|
|    2|       2.0| 254.1954634020619|
|    3|       3.0| 249.5825228971963|
|    4|       4.0|246.97514271428562|
|    5|       5.0|264.37037614150944|
|    6|       6.0| 295.1597153490566|
|    7|       7.0|243.64747528037387|
|    8|       8.0|195.25599892727263|
|    9|       9.0|206.09598121568627|
|   10|      10.0|205.93297300900903|
|   11|      11.0| 194.3172275445545|
|   12|      12.0| 199.3700942358491|
+-----+----------+------------------+
=======
## K-MEANS

Definition - What does K-Means Clustering mean?

K-means clustering is a simple unsupervised learning algorithm that is used to solve clustering problems. 

It follows a simple procedure of classifying a given data set into a number of clusters, defined by the letter "k," which is fixed beforehand. 

The clusters are then positioned as points and all observations or data points are associated with the nearest cluster, computed, adjusted and then the process starts over using the new adjustments until a desired result is reached.

K-means clustering has uses in search engines, market segmentation, statistics and even astronomy.

The algorithm:

K points are placed into the object data space representing the initial group of centroids.

Each object or data point is assigned into the closest k.

After all objects are assigned, the positions of the k centroids are recalculated.

Steps 2 and 3 are repeated until the positions of the centroids no longer move.


## CODE

```scala
//Las fuente de datos se encuentra en el repositorio: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Wholesalecustomersdata.csv

//1. Importar una simple sesión Spark.
import org.apache.spark.sql.SparkSession

//4. Importar la librería de Kmeans para el algoritmo de agrupamiento.
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._

//3. Cree una instancia de la sesión Spark
val spark = SparkSession.builder.appName("Practica").getOrCreate()

//7. Importar Vector Assembler y Vector
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}

//2. Utilice las lineas de código para minimizar errores
Logger.getLogger("org").setLevel(Level.ERROR)

//5. Carga el dataset de Wholesale Customers Data
val spark = SparkSession.builder().getOrCreate()
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale_customers_data.csv")

//6. Seleccione las siguientes columnas: Fres, Milk, Grocery, Frozen, Detergents_Paper,Delicassen y llamar a este conjunto feature_data
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

//8. Crea un nuevo objeto Vector Assembler para las columnas de caracteristicas como un conjunto de entrada, recordando que no hay etiquetas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//9. Utilice el objeto assembler para transformar feature_data
val traning = assembler.transform(feature_data)

//10. Crear un modelo Kmeans con K=3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(traning)

//11. Evalúe los grupos utilizando
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")

//12. ¿Cuáles son los nombres de las columnas?
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```

## RESULTADOS
```Sh
scala> :load Evaluacion.scala

Loading Evaluacion.scala...

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.clustering.KMeans

import org.apache.spark.ml.linalg.Vectors

import org.apache.log4j._

19/12/09 18:29:58 WARN SparkSession$Builder: Using an existing SparkSession; some configuration may not take effect.

spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@5cdf2c83

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}

spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@5cdf2c83

dataset: org.apache.spark.sql.DataFrame = [Channel: int, Region: int ... 6 more fields]

feature_data: org.apache.spark.sql.DataFrame = [Fresh: int, Milk: int ... 4 more fields]

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_b1698aee454a

traning: org.apache.spark.sql.DataFrame = [Fresh: int, Milk: int ... 5 more fields]

kmeans: org.apache.spark.ml.clustering.KMeans = kmeans_53225b5afc91

19/12/09 18:30:06 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS

19/12/09 18:30:06 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS

model: org.apache.spark.ml.clustering.KMeansModel = kmeans_53225b5afc91

warning: there was one deprecation warning; re-run with -deprecation for details

WSSSE: Double = 8.095172370767671E10

Within Set Sum of Squared Errors = 8.095172370767671E10

Cluster Centers: 

[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]

[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]

[35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
```
