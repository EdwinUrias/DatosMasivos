import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

df.show()
df.count
df.columns
df.filter($"Close" < 480 && $"High" < 480).show()
df.select(sum("High")).show()
df.("High")
df.select(max("High")).show()
df.select(month(df("Date"))).show()
df.select("High").first()
df.first()
df.describe()
df.sort()
df.select($"Close" < 500 && $"High" < 600).count()
df.printSchema()
df.select(year(df("Date"))).show()
df.select(min("High")).show()
df.filter($"High" > 480).count()
df.filter($"High"===484.40).show()
df.select(mean("High")).show()
df.select(corr("High", "Low")).show()