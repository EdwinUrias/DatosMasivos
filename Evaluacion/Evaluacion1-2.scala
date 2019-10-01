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