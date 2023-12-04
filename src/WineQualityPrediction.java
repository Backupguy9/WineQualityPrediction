import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {

    public static void main(String[] args) {
        SparkConf configuration = new SparkConf().setAppName("WineQualityPrediction").setMaster("local[*]");
        JavaSparkContext sparkContext = new JavaSparkContext(configuration);
        SparkSession session = SparkSession.builder().config(configuration).getOrCreate();

        String filePath = args[0];

        
        Dataset<Row> validationData = session.read().option("header", "true").option("delimiter", ";").csv(filePath);
        validationData.printSchema();
        validationData.show();

        
        for (String column : validationData.columns()) {
            if (!"quality".equals(column)) {
                validationData = validationData.withColumn(column, validationData.col(column).cast("double"));
            }
        }
        validationData = validationData.withColumnRenamed("quality", "target");

        
        VectorAssembler featureAssembler = new VectorAssembler().setInputCols(validationData.columns()).setOutputCol("featureSet");
        Dataset<Row> transformedData = featureAssembler.transform(validationData).select("featureSet", "target");
        transformedData.show();

        
        JavaRDD<LabeledPoint> rddData = convertToLabeledPoint(sparkContext, transformedData);

        
        RandomForestModel model = RandomForestModel.load(sparkContext.sc(), "s3://path/to/model");

        System.out.println("Model successfully loaded");

        
        JavaRDD<Double> predictionValues = model.predict(rddData.map(LabeledPoint::features));

        
        JavaRDD<Tuple2<Double, Double>> predictionAndLabels = rddData.map(point -> new Tuple2<>(point.label(), model.predict(point.features())));

        
        Dataset<Row> predictions = session.createDataFrame(predictionAndLabels, Tuple2.class).toDF("actualLabel", "predictedLabel");
        predictions.show();

        
        MulticlassMetrics evaluationMetrics = new MulticlassMetrics(predictions);
        double f1 = evaluationMetrics.fMeasure();
        System.out.println("F1 Score: " + f1);
        System.out.println("Confusion Matrix: " + evaluationMetrics.confusionMatrix());
        System.out.println("Precision: " + evaluationMetrics.weightedPrecision());
        System.out.println("Recall: " + evaluationMetrics.weightedRecall());
        System.out.println("Overall Accuracy: " + evaluationMetrics.accuracy());

        
        long errorCount = predictionAndLabels.filter(t -> !t._1.equals(t._2)).count();
        System.out.println("Error Rate = " + (double) errorCount / rddData.count());

        sparkContext.close();
    }

    private static JavaRDD<LabeledPoint> convertToLabeledPoint(JavaSparkContext context, Dataset<Row> dataFrame) {
        return dataFrame.javaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("target"));
            org.apache.spark.ml.linalg.Vector featuresVector = row.getAs("featureSet");
            double[] features = new double[featuresVector.size()];
            for (int i = 0; i < featuresVector.size(); i++) {
                features[i] = featuresVector.apply(i);
            }
            return new LabeledPoint(label, Vectors.dense(features));
        });
    }
}