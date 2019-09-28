import org.apache.spark.sql.SparkSession
//1
val spark = SparkSession.builder().getOrCreate()
//2
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//3
df.columns
//4
df.printSchema()
//5
df.head(5)
df.show(5)
//6 OPERACION PARA HACER UNA NUEVA COLUMNA

//7
val  = df.HVRatio("Year", year(df("Date")))
//8
//---------------------------------------------------------
//9
//RESPUESTA EN TEXTO
//10
df.select(max("Volume") && min("Volume")).show()
//11
//a)
df.filter($"Close" < 600).count()
//b)

//c)

//d)

//e)
