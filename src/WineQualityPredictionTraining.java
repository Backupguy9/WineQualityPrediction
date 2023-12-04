import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class WineQualityPredictionTraining {

    public static void main(String[] args) {
        // Initialize Spark configuration and context
        SparkConf conf = new SparkConf().setAppName("WineQualityModelTraining");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        
        String trainingDataPath = "C:\\Users\\daksh\\Downloads\\TrainingDataset.csv";
        Dataset<Row> trainingData = spark.read().format("csv").option("header", "true").option("delimiter", ";").load(trainingDataPath);

        
        for (String column : trainingData.columns()) {
            if (!column.equals("quality")) {
                trainingData = trainingData.withColumn(column, trainingData.col(column).cast("double"));
            }
        }
        trainingData = trainingData.withColumnRenamed("quality", "label");

        VectorAssembler assembler = new VectorAssembler().setInputCols(trainingData.columns()).setOutputCol("features");
        Dataset<Row> transformedTrainingData = assembler.transform(trainingData).select("features", "label");

        
        JavaRDD<LabeledPoint> trainingRDD = transformedTrainingData.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            double[] featuresArray = row.<org.apache.spark.ml.linalg.Vector>getAs("features").toArray();
            return new LabeledPoint(label, Vectors.dense(featuresArray));
        });

        
        int numClasses = 10; // Number of classes in 'label'
        int numTrees = 20; // Example value, adjust as needed
        String featureSubsetStrategy = "auto"; // Let the algorithm choose
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;
        int seed = 12345;

        
        RandomForestModel model = RandomForest.trainClassifier(trainingRDD, numClasses,
                new HashMap<>(), numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        
        String modelSavePath = "s3://path/to/model";         model.save(sc.sc(), modelSavePath);

        sc.close();
    }
}