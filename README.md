## Bank Marketing Campaign Success Predictor

### Deployed Link : [Website Link](https://bank-marketing-campaign-success-predictor.onrender.com)

### Problem Statement :
`The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. Our classification goal is to predict if the client will subscribe a term deposit (variable y).`

### Software and Tools Required

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
4. [Render](https://dashboard.render.com)

We can specify a location for the environment with the -p flag (or --prefix).

## Create a  new virtual environment:
`python -m venv myenv`

Note : 
1. Use python -m venv if you want a simple, python-specific environment.
2. Use conda create -p if you need a more full-featured environment manager with better support for non-python dependencies and are comfortable with Conda.


## Activate the virtual environment:

**Windows** : `myenv\Scripts\activate` <br>
**macOS\linux** : `source myenv/bin/activate`

### File/Folder Description :
1. `Templates` : It includes those file which is dynamically changing and used by render_template.
2. `Static` : It includes those file which is static in nature like css, js, images.

#### Dataset Source : [UCL Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

### Libraries Used 
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Imblearn (for oversampling)
6. Scikit Learn (Sklearn)
7. Flask


## Generate Requirements.txt file
`pip freeze > requirements.txt`

### Data Preprocessing Steps
1. Dropped features like emp.var.rate, cons.price.idx, pdays etc. because of correlation.
2. Used pipelines for preprocessing categorical and numerical columns using _One Hot Encoder_ and _Standard Scalar_.
3. Data is highly unbalanced (i.e 88% in favour of yes) , so balanced it by doing oversampling using _SMOTE_.

### Models Tried 
1. Logistic Regression
2. KNeighborsClassifier
3. Stochastic Gradient Descent(SGD) Classifier
4. DecisionTreeClassifier
5. RandomForestClassifier

### Accuracy Table

Below table is showing accuracy score of various models used in this task:

| Model                      | Precision | Recall | F1 Score | Accuracy (%) |
|----------------------------|-----------|--------|----------|--------------|
| Logistic Regression         | 0.43      | 0.82   | 0.56     | 85.41        |
| KNearest Neighbour          | 0.39      | 0.77   | 0.52     | 83.62        |
| SGD                         | 0.44      | 0.80   | 0.57     | 86.05        |
| Decision Tree               | 0.43      | 0.52   | 0.47     | 86.84        |
| Random Forest               | 0.58      | 0.49   | 0.53     | 90.13        |

### Best Model : Random Forest
### Note : 
  1. Further fine-tuned RandomForestClassifier due to high **accuracy** and **precision**.
  2. Here, according to me, `Precision is more crucial than Accuracy` because the bank is interested in knowing out of all 'Yes' predictions, how many were actually 'Yes'. Hence, we want        to reduce False Positives, and for this, we are focusing on Precision.

### For Home page :
1. HTML (for form creation)
2. CSS (for styling)
