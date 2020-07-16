


from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier


"""============================================================================================================="""

# create or get the spark session
spark = SparkSession.builder.appName("myapp").getOrCreate()

"""============================================================================================================="""

def ProcessData(df):  
        df= df.withColumn("label", df["Cancelled"].cast(IntegerType()))
       # categoricalColumns = ['Origin','Dest']
      
        #Categorical to Continuous/Ordinal/assigning the index
        categoricalColumns = ['Origin','Dest']
        for categoricalCol in categoricalColumns:
            stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index').fit(df)
            df=stringIndexer.transform(df)
         #One Hot Encoder
#        encoder = OneHotEncoderEstimator(inputCols=["OriginIndex", "DestIndex"],
#                                     outputCols=["categoryVec1", "categoryVec2"])
#        model = encoder.fit(df)
#        encoded = model.transform(df)
#        for categoricalCol in categoricalColumns:
#            stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index').fit(df)
#            df=stringIndexer.transform(df)
#    
        df = df.withColumn("YearInt", df["Year"].cast(IntegerType()))
        df = df.withColumn("MonthInt", df["Month"].cast(IntegerType()))
        df = df.withColumn("DayofMonthInt", df["DayofMonth"].cast(IntegerType()))
        df = df.withColumn("DayofWeekInt", df["DayOfWeek"].cast(IntegerType()))
        df = df.withColumn("DepTimeInt", df["DepTime"].cast(IntegerType()))
        df = df.withColumn("CRSDepTimeInt", df["CRSDepTime"].cast(IntegerType()))
        df = df.withColumn("ArrTimeInt", df["ArrTime"].cast(IntegerType()))
        df = df.withColumn("CRSArrTimeInt", df["CRSArrTime"].cast(IntegerType()))
       
        df = df.withColumn("ActualElapsedTimeInt", df["ActualElapsedTime"].cast(IntegerType()))
        df = df.withColumn("CRSElapsedTimeInt", df["CRSElapsedTime"].cast(IntegerType()))
        df = df.withColumn("ArrDelayInt", df["ArrDelay"].cast(IntegerType()))
        df = df.withColumn("DepDelayInt", df["DepDelay"].cast(IntegerType()))
        df = df.withColumn("DistanceInt", df["Distance"].cast(IntegerType()))
        #df= df.withColumn("label", df["Cancelled"].cast(IntegerType()))
#        encoder = OneHotEncoderEstimator(inputCols=["OriginIndex", "DestIndex"],
#                                     outputCols=["categoryVec1", "categoryVec2"])
#        model = encoder.fit(df)
#        encoded = model.transform(df)
#  
        assembler=VectorAssembler(inputCols=["YearInt","MonthInt","DayofMonthInt","DayofWeekInt","DepTimeInt","CRSDepTimeInt","ArrTimeInt","CRSArrTimeInt","ActualElapsedTimeInt","CRSElapsedTimeInt","ArrDelayInt","DepDelayInt","OriginIndex","DestIndex","DistanceInt"], outputCol="features")
        
           # assembler = VectorAssembler(inputCols=["YearInt","MonthInt","DayofMonthInt","DayofWeekInt","DepTimeInt","CRSDepTimeInt","ActualElapsedTimeInt","CRSElapsedTimeInt","ArrDelayInt","DepDelayInt","OriginIndex","DestIndex","DistanceInt"], outputCol="features")
        df = assembler.transform(df)
        return df

        """============================================================================================================="""

def Classify():
    print(" Training Step:1")
    train_df = spark.read.csv("/home/sunbeam/EndGame/Last_Try/airlines_Data.csv", header=True)
    train_df=train_df.drop('SrNo')
    
   # train_df=train_df.drop('FlightNum')
    train_df.printSchema()
    
    
    print("Training step:2")
    train_df = ProcessData(train_df)
    
    print("In Classify")

    rf = RandomForestClassifier(featuresCol = 'features',labelCol = 'label',maxDepth = 3,maxBins=304)
    model = rf.fit(train_df)
    
    """
    regressor = LogisticRegression()
    model = regressor.fit(train_df)
    """
    print("Training Done")
    
    print("Testing Step:1")
    test_df = spark.read.csv("/home/sunbeam/EndGame/Last_Try/myfile.csv", header=True)
    test_df.printSchema()
     #cols = test_df.columns
    
    print("Testing Step:2")
    test_df=ProcessData(test_df)
    test_df.printSchema()
    prediction = model.transform(test_df)
    prediction.select( 'label','rawPrediction', 'prediction', 'probability').show(10)
    
    return prediction.collect()[0]["prediction"]


app = Flask(__name__)

# routes
@app.route("/")
def serveRoot():
    return render_template("home.html")

@app.route("/process")
def processInput():
    #age = request.args["age"]
    #salary = request.args["salary"]
    #gender = request.args["gender"]
    
    Year = request.args["Year"]
    Month = request.args["Month"]
    DayofMonth = request.args["DayofMonth"]
    DayOfWeek = request.args["DayOfWeek"]
    DepTime = request.args["DepTime"]
    CRSDepTime= request.args["CRSDepTime"]
    ArrTime=request.args["ArrTime"]
    CRSArrTime= request.args["CRSArrTime"]
    #FlightNum=5206
    ActualElapsedTime= request.args["ActualElapsedTime"]
    CRSElapsedTime = request.args["CRSElapsedTime"]
    ArrDelay=request.args["ArrDelay"]
    DepDelay= request.args["DepDelay"]
    Origin=request.args["Origin"]
    Dest = request.args["Dest"]
    Distance= request.args["Distance"]
    #Cancelled=1
    print("hello")
    file = open("/home/sunbeam/EndGame/Last_Try/myfile.csv", "w")
    file.write("Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,ActualElapsedTime,CRSElapsedTime,ArrDelay,DepDelay,Origin,Dest,Distance,Cancelled\n{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,ActualElapsedTime,CRSElapsedTime,ArrDelay,DepDelay,Origin,Dest,Distance))
    file.close()
    
    result = Classify()
    
   # print("Year:{},Month:{},DayofMonth:{},DayOfWeek:{},DepTime:{},CRSDepTime:{},ArrTime:{},CRSArrTime:{},ActualElapsedTime:{},CRSElapsedTime:{},ArrDelay:{},DepDelay:{},Origin:{},Dest:{},Distance:{}".format(Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,FlightNum,ActualElapsedTime,CRSElapsedTime,ArrDelay,DepDelay, Origin, Dest, Distance))
    
    return render_template("result.html", result=result)

app.run()