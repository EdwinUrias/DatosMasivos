// Importacion de las librerias y apis de Logistic Regression
import org.apache.spark.ml.classification.LogisticRegression
//Importacion de la libreria de SparkSeccion
import org.apache.spark.sql.SparkSession
// Si requerimos la utilizacion del codigo de  Error reporting lo implementamos aunque regularmente es opcional esta accion.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
// Creamos la secion de spark. 
val spark = SparkSession.builder().getOrCreate()
// Utilice Spark para leer el archivo que se encuentra en un formato csv llamado Advertising.csv
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
//Visualizacion de los datos de nustro dataset y analizamos como nos son precentados.
data.show()
// Imprecion del Schema del DataFrame
data.printSchema()
//Despliegue los datos
// Impresion de un renglon de ejemplo 
data.head(1)
// creamos la variable colnames la cual contendra en un arreglo de string la informacion de la primera columna.
val colnames = data.columns
//creacion de la variable fristrow la cual contendra el contendido de la primera columna de datos.
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
//Preparar el DataFrame para Machine Learning
//Creamos una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click"
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
//    - Renombracion de la columna "Clicked on Ad" a "label"
val logregdata = timedata.select(data("Clicked on Ad").as("label") ,$"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
// Visualizacion del renombramiento de la columna Clicked on Ad a label.
logregdata.show()
// Importacion de las librerias y apis de VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
// Creacion de un nuevo objecto VectorAssembler llamado assembler para los feature las cuales seran las columnas restantes del dataset como un arreglo "Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"
val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features"))
// Utilizamos randomSplit para crear datos de entrenamineto con 70% y de prueba 30% con los que estara interactuando nuestro algoritmo 
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
//Configure un Pipeline
// Importacion de la libreria de Pipeline
import org.apache.spark.ml.Pipeline
// Creacion de un nuevo objeto de  LogisticRegression llamado lr
val lr = new LogisticRegression()
// Creacion de un nuevo  pipeline con los elementos: assembler la cual es nuestras factures, lr duentro objeto de LOgisticRegression
val pipeline = new Pipeline().setStages(Array(assembler, lr))
// creacion de la variable model la cual contendra el elemento de pipeline el cual contendra un ajuste (fit)  para el conjunto que nos encontramos entrenando
val model = pipeline.fit(training)
// Resultado del modelo con la trasformacion de los datos de prueba.
val results = model.transform(test)
results.show()
//Evaluacion del modelo
// Para la utilizacion de Metrics y Evaluation importamos la libreria de  MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// Convertimos los resutalos de la prueba (test) en RDD utilizando .as y .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
// Inicializamos el objeto MulticlassMetrics con el parametro predictionAndLabels.
val metrics = new MulticlassMetrics(predictionAndLabels)
// Impresion de la matrix de confucion
println("Confusion matrix:")
println(metrics.confusionMatrix)
// Imprimimos la exactitud de nuestra predicion la cual es de 0.97 la cual es muy buena.
metrics.accuracy