import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="CAPSTONE PROJECT", layout="wide")
data = pd.read_csv('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/updated1_survey.csv')
page = st.sidebar.radio("Select a Page", ("Home", "Exploratory Data Analysis", "Classification", "Regression", "Unsupervised"))

if page == "Home":
    st.title("CAPSTONE PROJECT")
    st.header("About This Project")
    st.divider()
    st.subheader("Welcome! everyone I built this website to show my machine learning project. As we know in today's era mental health is big issue among the working professionals.")
    st.markdown('''
### What does this project showcase?

- Exploratory Data Analysis- Showing the numbers in the form of interactive charts and graphs.

- Classification- Predicting whether a person is likely to seek mental health treatment or not.

- Regression- Predicting the age of a respondent based on various features.

- Unsupervised Learning- Grouping tech workers into different mental health Personas using clustering.
''')

    st.subheader("Dataset Overview")
    st.code(data.head)
    st.write('To Download full data click on button below.')
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download This Dataset in Csv File",
        data=csv,
        file_name="updated1_survey.csv",
        mime='text/csv'
    )

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    column_selection=st.radio("Select EDA which you want to see",['Age','Gender','Country','Self Employed',
                                                      'Family History','Treatment History','Work Interfere',
                                                      'Heatmap(Correlation Matrix)'])
    if column_selection == 'Age':
        st.title("Age Distribution of Respondents")
        st.divider()
        st.write('This bar helps us to understand which age groups have more respondents.')
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-07%20202225.png')
        st.markdown("Here we clearly see most of age groups fall between 26-33. This shows that tech force is largely young.This column in dataset is important because in today's most mental health patients from young age group.")
    elif column_selection == 'Gender':
        st.title("Gender Distribution ")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-11%20170747.png')
        st.markdown('''- In this we clearly see most of workers are male which are 78.5%.
- 19.4% tech workers are female.
- 2.1% tech workers are transgender.
''')
    elif column_selection == 'Country':
        st.title("Country Distribution")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205533.png')
        st.markdown('''- In this we clearly see top five countries are
- United States- 70% people are from United States.
- United Kingdom- 17% people live in United Kingdom.
- Canada- 7% people live in Canada.
- Germany- 4% people live in Germany.
- Netherlands- 2% people live in Netherlands.''')
    elif column_selection == 'Self Employed':
        st.title("Self Employed Distribution")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205647.png')
        st.markdown('''
- In this pie plot we clearly see that 87.0% people are doing job.
- 11.6% people are self employed.
- 1.4% people are not verified yet.
''')
    elif column_selection == 'Treatment History':
        st.title("Treatment History")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205703.png')
        st.markdown('''### In above distribution we clearly see. 
- 50.6% people are takes treatment in past. 
- 49.4% people does not takes treatment in past.''')
    elif column_selection == 'Work Interfere':
        st.title("Work Interfere")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-12%20084851.png')
        st.markdown('''### In above let us see meaning of some words. 
- Sometimes means work interfere sometimes affect mental health not always.
- Not Verified means answer not clear or missing.
- Never means work never affect mental health.
- Rarely means very few times work affect mental health.
- Often means work affect many times.''')
        st.markdown('''
- 6-25 employees high count
- 26-100 employees high count
- More than 1000 employees high count
- 100-500 employees medium count
- 1-5 employees medium count
- 500-1000 employees lowest count
''')
    elif column_selection == 'Family History':
        st.title("Family History")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205654.png')
        st.markdown('''### Here in above pie chart we clearly see.
- 39.1% of working professionals have mental issue in past in their family.
- 60.9 % of professionals does not have in mental health in past in their family.''')
    elif column_selection == 'Heatmap(Correlation Matrix)':
        st.title("Heatmap(Correlation Matrix)")
        st.divider()
        st.image('https://raw.githubusercontent.com/Ravinder1191/CAPSTONE-PROJECT/main/MachineLearningProject/EDA_CSV/Screenshot%202025-08-03%20205725.png')
        st.markdown('''### Correlation Matrix
- This graph show how each column relate to other columns in numbers between -1 and 1
- 1 means strong positive relation both values go up together.
- 0 means no relation.
- -1 means strong negative relation one goes up other goes down.
- Dark color means high relation.
- Light color means low relation.
- We use this to see which features connect more or less in data.''')
elif page == 'Classification':
    st.title("Classification")
    st.divider()
    st.subheader("Predicting Whether a Person is Likely To Seek Mental Health Treatment or Not.")
    st.markdown('''
## Models Used?
- Gradient Boosting Classifier  
- Decision Tree Classifier
## What is Gradient Boosting Classifier?
- It combines many weak learners(decision trees).
## Working?
- It build model in stages.Each new tree tries ot correct errors of last
- final prediction based on combination of all trees,
## accuracy
    0.8269841269841269
## roc auc
    0.8201507542765917
## f1 score
    0.8529014844804319  
## confusion matrix
    [[224  75]
    [ 42 289]]
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
- It makes decisions by splitting the data based on feature values using tree like structure.
## Working?
- Starts from top which is called root node.
- Based on result splits data.
- Continues splitting until reaches final decision.
## accuracy
    0.8142857142857143
## roc auc
    0.8111378310380017
## f1 score
    0.8316546762589928
## confusion matrix
    [[224  75]
    [ 42 289]]
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
- It combines both lasso and ridge regression regularization which are also called L1 and L2 Regularization.
- Lasso:Helps with feature selection.
- Ridge:Helps to reduce overfitting.
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
- Starts with each data point as its own cluster.
- In each step, It merges the two closest clusters.
- Continues merging until the required number of clusters is formed.
- Distance between clusters is calculated using linkage methods like average or euclidean distance formula.
## Note:Whole linkage formed is called dendrogram
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



