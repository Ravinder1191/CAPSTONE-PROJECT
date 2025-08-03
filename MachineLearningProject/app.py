# streamlit run C:\Users\Ravin\PycharmProjects\MachineLearningProject\app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CAPSTONE PROJECT", layout="wide")
data = pd.read_csv("https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/updated1_survey.csv")
page = st.sidebar.selectbox("Select a Page", ("Home", "Exploratory Data Analysis", "Classification", "Regression", "Unsupervised"))

if page == "Home":
    st.title("CAPSTONE PROJECT")
    st.header("About This Project")
    st.divider()
    st.subheader("Welcome! everyone I Built this website to show my Machine Learning Project. As we know in today's era mental health is big issue among the working professionals.")

    st.markdown('''
### What Does This Project Showcase?

- Exploratory Data Analysis- Showing The Numbers In The Form Of Interactive Charts and Graphs.

- Classification- Predicting Whether a Person is Likely To Seek Mental Health Treatment or Not.

- Regression- Predicting The Age Of a Respondent Based on Various Features.

- Unsupervised Learning- Grouping Tech Workers Into Different Mental Health Personas Using Clustering.
''')

    st.subheader("Dataset Overview")
    st.dataframe(data)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download This Dataset in Csv File",
        data=csv,
        file_name="updated1_survey.csv",
        mime='text/csv'
    )

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.divider()

elif page == "Classification":
    st.title("Classification")              #[[205  94]
                                            #[ 15 316]]')
    st.markdown('''
### Model used for classification
- Gradient Boosting Classifier - A Gradient Boosting Classifier Is a Machine Learning Algorithm Used For Classification Tasks, Belonging To The Family Of Ensemble methods. It Sequentially Combines Multiple Typically Decision Trees, To create a Strong Predictive Model.
- Features Used- work interfere and family history 
### How features are selected?
- Using heatmap first choose the target row or column and see it's corresponding row and columns values less 0.30 ignore values greater than 0.30 are best. 
### Scores
# Decision Tree Classifier
- A decision Tree classifier Is a Supervised Machine Learning Algorithm Used For Classification tasks. It Employees a Tree-Like structure To Model Decisions and Their Possible Consequences, Ultimately Classifying Data Instances Into Discrete Categories. 
''')
    st.title('Model Performance')
    st.divider()
    st.markdown('''
## Accuracy: 
    0.8269841269841269  
## roc auc: 
    0.8201507542765917  
## f1 score: 
    0.8529014844804319

    ''')
    classification_data = {
        "Class": [0, 1, "accuracy", "macro avg", "weighted avg"],
        "Precision": [0.93, 0.77, "", 0.85, 0.85],
        "Recall": [0.69, 0.95, "", 0.82, 0.83],
        "F1-Score": [0.79, 0.85, "", 0.82, 0.82],
        "Support": [299, 331, 630, 630, 630]
    }

    classification_report = pd.DataFrame(classification_data)
    st.table(classification_report)

    st.title('Model Performance of Decision Tree Classifier')
    st.markdown('''
## accuracy:
    0.8142857142857143
## roc auc:
    0.8111378310380017
## f1 score:
    0.8316546762589928    
    ''')

    classification_data = {
        "Class": [0, 1, "accuracy", "macro avg", "weighted avg"],
        "Precision": [0.84, 0.79, "", 0.82, 0.82],
        "Recall": [0.75, 0.87, "", 0.81, 0.81],
        "F1-Score": [0.79, 0.83, 0.81, 0.81, 0.81],
        "Support": [299, 331, 630, 630, 630]
    }
    classification_report = pd.DataFrame(classification_data)
    st.table(classification_report)

elif page == "Regression":
    st.title("Regression")
    st.divider()
    st.markdown('''
### Model used to predict age
- Elastic Net Regression Model
### What is Elastic Net Regression?
- Elastic Net Regression is a Regularized Linear Regression Model That Combines The Penalties of Both Lasso (L1) and Ridge (L2) Regularization.It is Designed to Address Limitation of Lasso and Ridge
    ''')
    st.title('Model Scores')
    st.divider()
    st.markdown('''
## R2 Score:0.0449
## MSE:
    50.43
## MAE:
    5.51
- Here below is Graphical Representation.
    ''')

elif page == "Unsupervised":
    st.title('Unsupervised Learning')
    st.divider()
    st.subheader('Model used to group tech workers')
    st.markdown('''
## Agglomerative Clustering Is Used To Group Tech Workers Into Different Mental Health Personas
-  It Is type of Hirerchal Clustering In This We Assume All Datapoints In Indiviual Cluster and Then start Grouping Them With Nearest Datapoint and Linkage is Established is Dendrogram
- In Below Image How Dendrogram Looks Like Shown Below  
    ''')

    st.subheader('Unsupervised Learning Score ')
    st.markdown('''
## Agglomerative Clustering Silhouette Score: 
    0.1533
- Here Below Is Graphical Representation.
    ''')


