<h2>Simple Linear Regression</h2>

<p>
<strong>Simple Linear Regression (SLR)</strong> is a statistical and machine learning technique
used to model the <strong>linear relationship</strong> between two variables:
</p>

<ul>
  <li><strong>One independent variable</strong> — often called feature (X)</li>
  <li><strong>One dependent variable</strong> — target (Y)</li>
</ul>

<p>
It assumes that changes in <strong>X</strong> result in proportional changes in <strong>Y</strong>,
and fits a straight line through the observed data to make predictions.
</p>

<hr>

<h2>🧠 Model Equation</h2>

<p>The model is represented by the equation:</p>

<p style="font-size:18px;">
  <strong>y = β<sub>0</sub> + β<sub>1</sub>x + ε</strong>
</p>

<h4>Where:</h4>
<ul>
  <li><strong>y</strong> = dependent variable (value we want to predict)</li>
  <li><strong>x</strong> = independent variable (input)</li>
  <li><strong>β<sub>0</sub></strong> = intercept (predicted value of y when x = 0)</li>
  <li><strong>β<sub>1</sub></strong> = slope/coefficient (how strongly x affects y)</li>
  <li><strong>ε</strong> = error term (difference between actual and predicted values)</li>
</ul>

<p>
👉 The goal is to choose values of <strong>β<sub>0</sub></strong> and <strong>β<sub>1</sub></strong>
so that the predicted line fits the data as closely as possible.
</p>

<hr>

<h2>🧮 How Simple Linear Regression Works</h2>

<h3>1. Data Collection</h3>
<p>
Collect paired data samples:
<strong>(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)</strong>
</p>

<h3>2. Fit the Best Line — Least Squares</h3>
<p>
The most common method used is <strong>Ordinary Least Squares (OLS)</strong>.
This method finds the line that minimizes the
<strong>sum of squared differences</strong> between actual and predicted Y values.
</p>

<p>Steps involved:</p>
<ul>
  <li>Calculate mean of X (<strong>x̄</strong>) and mean of Y (<strong>ȳ</strong>)</li>
  <li>Compute slope (β<sub>1</sub>) and intercept (β<sub>0</sub>) using formulas</li>
</ul>

<p>
After fitting, the regression line approximates the trend present in the data.
</p>

<hr>

<h2>📈 Interpretation of the Model</h2>

<h3>📌 Slope (β<sub>1</sub>)</h3>
<ul>
  <li>If β<sub>1</sub> &gt; 0 → Positive relationship (X increases → Y increases)</li>
  <li>If β<sub>1</sub> &lt; 0 → Negative relationship</li>
  <li>The magnitude shows how much Y changes for a unit change in X</li>
</ul>

<h3>📌 Intercept (β<sub>0</sub>)</h3>
<ul>
  <li>Predicted value of Y when X = 0</li>
  <li>Acts as a baseline prediction</li>
</ul>

<h3>📌 Error Term (ε)</h3>
<p>
Represents the portion of Y that is not explained by the linear relationship.
</p>

<hr>

<h2>🛠️ Typical Implementation in a Project</h2>

<p>
The <strong>SLR_Project</strong> repository contains the following files:
</p>

<ul>
  <li><strong>app.py</strong> — Python script or web application (Flask/Streamlit)</li>
  <li><strong>slr.pkl</strong> — Trained Simple Linear Regression model saved using pickle</li>
  <li><strong>requirements.txt</strong> — List of required Python dependencies</li>
</ul>

<hr>

<h2>🧩 app.py – Project Workflow</h2>

<p>The <code>app.py</code> file typically performs the following steps:</p>

<ol>
  <li>Loads the trained model (<code>slr.pkl</code>)</li>
  <li>Accepts input data from the user or frontend</li>
  <li>Applies the regression model to predict Y from X</li>
  <li>Displays the prediction output</li>
</ol>

<h4>Sample Code Structure:</h4>

<pre>
<code>
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = pickle.load(open('slr.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    x_value = float(request.form['x'])
    y_pred = model.predict([[x_value]])
    return jsonify({'predicted_value': y_pred[0]})

if __name__ == "__main__":
    app.run(debug=True)
</code>
</pre>

<p>This code:</p>
<ul>
  <li>Loads the saved regression model</li>
  <li>Predicts output Y for given input X</li>
  <li>Returns the prediction as a response</li>
</ul>

<hr>

<h2>🧪 Performance and Evaluation</h2>

<p>
A complete Simple Linear Regression project usually includes:
</p>

<ul>
  <li>Splitting data into training and testing sets</li>
  <li>Training the model using training data</li>
  <li>Evaluating performance using metrics</li>
</ul>

<h4>Common Metrics:</h4>
<ul>
  <li><strong>Mean Squared Error (MSE)</strong></li>
  <li><strong>R² Score</strong> (explains variance in data)</li>
</ul>

<h4>Example Evaluation Code:</h4>

<pre>
<code>
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
</code>
</pre>

<hr>

<h2>📌 What This Project Demonstrates</h2>

<ul>
  <li>✔ Data-driven prediction using Simple Linear Regression</li>
  <li>✔ Use of Python and ML libraries (scikit-learn)</li>
  <li>✔ Saving and reusing trained models (<code>slr.pkl</code>)</li>
  <li>✔ Deploying predictions via a Python script or web application</li>
</ul>

<p>
This project represents a complete <strong>end-to-end mini machine learning workflow</strong>:
training → saving → deploying → predicting.
</p>

<hr>

<h2>📌 Project Summary</h2>

<p>
Simple Linear Regression models the relationship between one input feature and one target variable.
It assumes a straight-line relationship and finds optimal coefficients that minimize prediction error.
This project demonstrates practical implementation and deployment of a Simple Linear Regression model.
</p>
