# DatosMasivos
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
```
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
