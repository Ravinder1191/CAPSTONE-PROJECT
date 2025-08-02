# streamlit run C:\Users\Ravin\PycharmProjects\MachineLearningProject\app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CAPSTONE PROJECT", layout="wide")
data = pd.read_csv(r"C:\Users\Ravin\Downloads\updated1_survey.csv")
page = st.sidebar.selectbox("Select a Page", ("Home", "Exploratory Data Analysis", "Classification", "Regression", "Unsupervised"))

if page == "Home":
    st.title("CAPSTONE PROJECT")
    st.header("About This Project")
    st.divider()
    st.subheader("Welcome! everyone I Built this website to show my Machine Learning Project. As we know in today's era mental health is big issue among the working professionals.")

    st.markdown('''
### What Does This Project Showcase?

- Exploratory Data Analysis- Showing the numbers in the form of interactive charts and graphs.

- Classification- Predicting whether a person is likely to seek mental health treatment or not.

- Regression- Predicting the **age** of a respondent based on various features.

- Unsupervised Learning- Grouping tech workers into different mental health personas** using clustering.
''')

    st.subheader("Dataset Overview")
    st.dataframe(data)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download This Dataset in csv file",
        data=csv,
        file_name="updated1_survey.csv",
        mime='text/csv'
    )

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.divider()
    st.subheader("In above pie chart we clearly see that most of working professionals are male which are around 78.5% in tech companies 19.4% are female and rest are others")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213705.png")
    st.subheader("These are top 5 Countries Having Most number of working employees")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213729.png")
    st.subheader("There are more all columns analysis below of Dataset")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213754.png")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213800.png")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213808.png")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213816.png")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213824.png")
    st.subheader("Heatmap Distribution of all categories")
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 213846.png")

elif page == "Classification":
    st.title("Classification")              #[[205  94]
                                            #[ 15 316]]')
    st.markdown('''
### Model used for classification
- Gradient Boosting Classifier - A Gradient Boosting Classifier is a machine learning algorithm used for classification tasks, belonging to the family of ensemble methods. It sequentially combines multiple "weak learners," typically decision trees, to create a strong predictive model.
- Features Used- work interfere and family history 
### How features are selected?
- Using heatmap first choose the target row or column and see it's corresponding row and columns values less 0.30 ignore values greater than 0.30 are best. 
### Scores
# Decision Tree Classifier
- A decision tree classifier is a supervised machine learning algorithm used for classification tasks. It employs a tree-like structure to model decisions and their possible consequences, ultimately classifying data instances into discrete categories. 
''')
    st.title('Model Performance')
    st.divider()
    st.markdown('''
## accuracy: 
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
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-02 160047.png")

elif page == "Regression":
    st.title("Regression")
    st.divider()
    st.markdown('''
### Model used to predict age
- Elastic Net Regression Model
### What is Elastic Net Regression?
- Elastic net regression is a regularized linear regression model that combines the penalties of both Lasso (L1) and Ridge (L2) regularization.It is Designed to address limitation of lasso and Ridge
    ''')
    st.title('Model Scores')
    st.divider()
    st.markdown('''
## R2 Score:0.0449
## MSE:
    50.43
## MAE:
    5.51
- Here below is graphical representation.
    ''')
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 104641.png")
elif page == "Unsupervised":
    st.title('Unsupervised Learning')
    st.divider()
    st.subheader('Model used to group tech workers')
    st.markdown('''
## Agglomerative Clustering is used to group tech workers into different mental health personas
-  it is type of Hirerchal clustering in this we assume all datapoints in indiviual cluster and then start grouping them with nearest datapoint and linkage is established is dendrogram
- in below image how dendrogram looks like shown    
    ''')
    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-01 110923.png")
    st.subheader('Unsupervised Learning Score ')
    st.markdown('''
## Agglomerative Clustering Silhouette Score: 
    0.1533
- here below is graphical representation.
    ''')

    st.image(r"C:\Users\Ravin\OneDrive\Pictures\Screenshots\Screenshot 2025-08-02 162814.png")

