using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace FirstMLModel
{
    class Program
    {
        //Define data structure. This is clases to hold the data
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;
            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        //IrisPrediction is the result returned prediction by the model
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            //Create a PipeLine and Load the Data
            var pipeline = new LearningPipeline();
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            //Transform the Data
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            //Add Learning/Training Algorithm
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            //Train the Model
            var model = pipeline.Train<IrisData, IrisPrediction>();

            //Use the Model to make Prediction
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 0.3f,
                SepalWidth = 0.6f,
                PetalLength = 1.2f,
                PetalWidth = 1.1f
            });
            Console.WriteLine($"Pridicted flower class is : {prediction.PredictedLabels}");
            Console.Read();
        }
    }
}
