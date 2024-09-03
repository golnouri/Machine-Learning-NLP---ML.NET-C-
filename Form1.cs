using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace WinFormsApp1
{
    public partial class Form1 : Form
    {
        public class Comments
        {
            public int Id { get; set; }
            public string Name { get; set; }
            public string CommentsText { get; set; }
            public bool IsPositive { get; set; }
        }

        public class CommentData
        {
            public string CommentsText { get; set; }
            public bool Label { get; set; }
        }

        public class CommentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool IsPositive { get; set; }
        }

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            List<Comments> com = new List<Comments>
            {
                new Comments { Id = 1, Name = "علی", CommentsText = "نظرم چندان خوب نبود", IsPositive = false },
                new Comments { Id = 2, Name = "مهدی", CommentsText = "نظرم بد نبود میشه بهترشم کرد", IsPositive = true },
                new Comments { Id = 3, Name = "نگین", CommentsText = "خیلی بد", IsPositive = false },
                new Comments { Id = 4, Name = "نگار", CommentsText = "واقعا بد بود ...", IsPositive = false },
                new Comments { Id = 5, Name = "سید", CommentsText = "اصلا خوشم نیومد. منفی", IsPositive = false },
                new Comments { Id = 6, Name = "زهرا", CommentsText = "خیلی خوب بود", IsPositive = true },
                new Comments { Id = 7, Name = "سارا", CommentsText = "ازش راضی بودم", IsPositive = true },
                new Comments { Id = 8, Name = "حسین", CommentsText = "بدک نبود", IsPositive = true },
                new Comments { Id = 9, Name = "محمد", CommentsText = "عالی بود", IsPositive = true },
                new Comments { Id = 10, Name = "فرشته", CommentsText = "کاملاً ضعیف", IsPositive = false }
            };

            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(com.Select(c => new CommentData { CommentsText = c.CommentsText, Label = c.IsPositive }));

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(CommentData.CommentsText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            var model = pipeline.Fit(data);

            // اضافه کردن نظرات جدید بدون IsPositive
            var newComments = new List<CommentData>
            {
                new CommentData { CommentsText = "این بهترین چیزی است که تا حالا دیدم!" },
                new CommentData { CommentsText = "کاملاً بی فایده بود." }
            };

            foreach (var newComment in newComments)
            {
                var predictionEngine = mlContext.Model.CreatePredictionEngine<CommentData, CommentPrediction>(model);
                var prediction = predictionEngine.Predict(newComment);
                MessageBox.Show($"Comment: {newComment.CommentsText}\nIs Positive: {prediction.IsPositive}");
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form2 frm2 = new Form2();
            frm2.ShowDialog();
        }
    }
}
