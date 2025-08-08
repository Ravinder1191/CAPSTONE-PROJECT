import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="CAPSTONE PROJECT", layout="wide")
data = pd.read_csv('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/updated1_survey.csv')
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
    column_selection=st.selectbox("Select EDA which you want to see",['Age','Gender','Country','State','Self Employed',
                                                      'Family History','Treatment History','Work Interfere',
                                                      'Number of Employees','Heatmap(Correlation Matrix)'])
    if column_selection == 'Age':
        st.title("Age Distribution of Persons.")
        st.write('This bar helps us to understand which age groups have more respondents.')
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-07%20202225.png')
        st.markdown('''Here we clearly see most of age groups are''')
elif page == "Classification":
    st.title("Classification")
    st.divider()
    st.write("Predicting Whether a Person is Likely To Seek Mental Health Treatment or Not.")
    st.markdown('''
## Models Used?
- Gradient Boosting Classifier  
- Decision Tree Classifier
## What is Gradient Boosting Classifier?
- It Combines Many Weak (Decision Trees).
## Working?
- It Build Model in stages.Each New Tree tries ot correct errors of last
- Final Prediction Based on Combination of all trees,
## accuracy
    0.8269841269841269
## roc auc
    0.8201507542765917
## f1 score
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

    st.markdown('''
## What is Decision Tree Classifier?
- It makes Decisions by Splitting The Data Based on Feature Values Using Tree like Structure.
## Working?
- Starts from top which is called root node.
- Based on result splits data.
- Continues Splitting Until reaches Final Decision.
## accuracy
    0.8142857142857143
## roc auc
    0.8111378310380017
## f1 score
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
    st.subheader("Decision Tree of Model")
    st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205829.png')

elif page == "Regression":
    st.title("Regression")
    st.divider()
    st.subheader("Predicting Age of Persons")
    st.markdown('''
## Which Model is Used?
- Elastic Net Regression.
## Working?
- It Combines Both Lasso and Ridge Regression regularization which are also called L1 and L2 Regularization.
- Lasso:Helps with Feature Selection.
- Ridge:Helps to Reduce Overfitting.
''')
    st.title("Model Scores")
    st.markdown('''
## R2 Score: 
    0.0449  
## MSE: 
    50.43  
## MAE: 
    5.51  

- Below is the graphical representation of predictions vs actual values.
''')
    st.image("https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205817.png")
elif page == "Unsupervised":
    st.title('Unsupervised Learning')
    st.divider()
    st.subheader('Group Tech Workers in Mental Health Personas')
    st.markdown('''
## Which Model is Used?
- Agglomerative Clustering.
## Working?
- Starts with each Data Point as its own Cluster.
- In each step, It merges the two Closest Clusters.
- Continues merging until the required number of clusters is formed.
- Distance Between Clusters is Calculated using Linkage methods like average or Euclidean distance Formula.
## Note:Whole Linkage Formed is called Dendrogram
''')
    st.subheader('Dendrogram of Model')
    st.divider()
    st.image("https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205746.png")
    st.subheader('Unsupervised Learning Score')
    st.markdown('''
## Agglomerative Clustering Silhouette Score: 
    0.1533
- Below is the Graphical Representation using 3D PCA.
''')
    features = ['family_history', 'treatment', 'work_interfere', 'remote_work', 'coworkers',
                'supervisor', 'no_employees', 'leave']
    encoder = OrdinalEncoder()
    data[features] = encoder.fit_transform(data[features])

    data[features] = data[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(scaled_data)

    pca_3d = PCA(n_components=3)
    reduced_3d = pca_3d.fit_transform(scaled_data)
    df = pd.DataFrame(reduced_3d, columns=["PCA1", "PCA2", "PCA3"])
    df["Cluster"] = agg_labels
    plot_3d = px.scatter_3d(df, x="PCA1", y="PCA2", z="PCA3",
                            color=df["Cluster"].astype(str),
                            title="3D PCA Visualization (Agglomerative Clustering)")
    st.subheader('2D Visualization of PCA')
    st.image("https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205805.png")
    st.subheader('3D Visualization of PCA')
    st.plotly_chart(plot_3d)
