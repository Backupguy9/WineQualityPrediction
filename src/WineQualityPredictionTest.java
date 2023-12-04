import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPredictionTest {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("winequality").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        String path = args[0];
        
        
        Dataset<Row> val = spark.read().format("csv").option("header", "true").option("sep", ";").load(path);
        val.printSchema();
        val.show();

        
        for (String colName : val.columns()) {
            if (!colName.equals("quality")) {
                val = val.withColumn(colName, val.col(colName).cast("float"));
            }
        }
        val = val.withColumnRenamed("quality", "label");

        // Getting the features and label separately and converting them to a numpy array
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(val.columns()).setOutputCol("features");
        Dataset<Row> df_tr = assembler.transform(val).select("features", "label");
        df_tr.show();

                JavaRDD<LabeledPoint> dataset = toLabeledPoint(sc, df_tr);

        
        RandomForestModel RFModel = RandomForestModel.load(sc.sc(), "/winepredict/trainingmodel.model/");

        System.out.println("Model loaded successfully");

        
        JavaRDD<Double> predictions = RFModel.predict(dataset.map(LabeledPoint::features));

        
        JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = dataset.map(lp -> new Tuple2<>(lp.label(), RFModel.predict(lp.features())));

        
        Dataset<Row> labelPred = spark.createDataFrame(labelsAndPredictions, Tuple2.class).toDF("label", "Prediction");
        labelPred.show();

        
        MulticlassMetrics metrics = new MulticlassMetrics(labelPred);
        double F1score = metrics.fMeasure();
        System.out.println("F1-score: " + F1score);
        System.out.println(metrics.confusionMatrix());
        System.out.println(metrics.weightedPrecision());
        System.out.println(metrics.weightedRecall());
        System.out.println("Accuracy: " + metrics.accuracy());

        
        long testErr = labelsAndPredictions.filter(lp -> !lp._1().equals(lp._2())).count();
        System.out.println("Test Error = " + (double) testErr / dataset.count());

        sc.close();
    }

    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> df) {
        return df.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            Vector features = Vectors.dense(row.<Double>getAs("features"));
            return new LabeledPoint(label, features);
        });
    }
}
