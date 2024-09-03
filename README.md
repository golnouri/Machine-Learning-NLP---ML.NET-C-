In this example, user comments are received as textual data (`CommentsText') and this goal is to use a machine model, these comments are divided into two categories: positive (`IsPositive = true') or negative (`IsPositive = false'). are classified.

### General code steps:

1. **Definition of data:**
 - You have a class called "Comments" that contains information such as "Id", "Name", "CommentsText" (text of the comment) and "IsPositive" (positive or negative of the comment).
 - Sample data from user comments are stored in a ``list<comments>''.

2. **Creating a car design model:**
 - First, it is converted to the format that the car model works with (CommentData).
 - An ``MLContext'' is then created to handle the machine's processing.
 - Data is loaded enumerably from the comment list and is ready for modeling.
 - In the next step, a ``pipeline'' is created, which includes two steps:
 - **FeaturizeText:** Convert the text of the comments to numeric features that models process.
 - **Trainer:** A classification algorithm (here `SdcaLogisticRegression`) that trains the model to distinguish between positive and negative comments.

3. **Prediction of new comments:**
 - Two new comments are added to the data whose "IsPositive" is not known.
 - A ``PredictionEngine'' is created for each new comment to predict whether the comment is positive or negative using the trained model.
 - The prediction result is displayed to the user using "MessageBox".

### Result:
This code allows you to create a simple machine model that divides user comments into positive and negative categories. This is a simple example of using NLP in a car.

Mojtaba Golnouri (golnouri@gmail.com)
Good Luck.
