import streamlit as st
st. set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Basic preprocessing required for all the models.  
def preprocessing(df):
    
    df = df.drop('Unnamed: 32', axis=1)
    
    cols = ['radius_worst', 
         'texture_worst', 
         'radius_mean',
         'perimeter_worst', 
         'compactness_mean',
         'symmetry_mean',
         'fractal_dimension_mean',
         'fractal_dimension_se',
         'radius_worst',
         'radius_se',
         'perimeter_se',
         'smoothness_se',
         'perimeter_mean',
         'smoothness_mean',
         'area_worst',
         'compactness_worst',
         'compactness_se',
         'concave points_worst']
    df = df.drop(cols,axis=1)
    


    cols = [
        'concave points_mean', 
        'concave points_se']
    df = df.drop(cols, axis=1)
    

    # Assign x and y
    x = df.iloc[:,2:].values
    y = df.iloc[:,1].values
    
    return x, y


# Training KNN Classifier
@st.cache(allow_output_mutation=True)
def Knn_Classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, knn    


# Training Logistic Regression
@st.cache(allow_output_mutation=True)
def logisticRegression(x_train, x_test, y_train, y_test):
    log = LogisticRegression(random_state=1)
    log.fit(x_train,y_train)
    y_pred = log.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    

    return score, report, log


# Training Support Vector Machine
@st.cache(allow_output_mutation=True)
def svm(x_train, x_test, y_train, y_test):
    svc_linear = SVC(kernel = 'linear', random_state = 1)
    svc_linear.fit(x_train, y_train)
    y_pred = svc_linear.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    

    return score, report, svc_linear


# Training Decision Tree
@st.cache(allow_output_mutation=True)
def Decision_Tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, tree

# Training Random Forest Classifier
@st.cache(allow_output_mutation=True)
def Random_Forest(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators= 10, criterion= 'entropy', random_state = 0)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, forest

def accept_user_data():
    
    area_mean = st.number_input("Enter the area_mean: ")
    texture_mean = st.number_input("Enter the texture_mean ")
    concavity_mean = st.number_input("Enter the concavity_mean: ")
    texture_se = st.number_input("Enter the texture_se: ")
    area_se = st.number_input("Enter the area_se: ")
    concavity_se = st.number_input("Enter the concavity_se: ")
    symmetry_se = st.number_input("Enter the symmetry_se: ")
    smoothness_worst = st.number_input("Enter the smoothness_worst: ")
    concavity_worst = st.number_input("Enter the concavity_worst: ")
    symmetry_worst = st.number_input("Enter the symmetry_worst: ")
    fractal_dimension_worst = st.number_input("Enter the fractal_dimension_worst: ")
    results = [texture_mean,area_mean,concavity_mean,texture_se,area_se,concavity_se,symmetry_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst]
    
 
    sample_data = np.array(results).reshape(1,-1)

    return sample_data, results



def main():
    tit1,tit2 = st.columns((4, 1))
    tit1.markdown("<h1 style='text-align: center;'><u>Machine Learning for Cancer Predictions</u> </h1>",unsafe_allow_html=True)
    st.sidebar.title("Dataset and Classifier")

    df = pd.read_csv("BreastCancer.csv")
    x,y = preprocessing(df)
    LE=LabelEncoder()
    y = LE.fit_transform(y)

    # Splitting x,y into Training & Test set.
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101) 

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Select dataset name
    dataset_name=st.sidebar.selectbox("Select Dataset: ",["NONE","Breast Cancer Dataset"])
    if (dataset_name=="Breast Cancer Dataset"):
        st.header("Breast Cancer Detection WebApp")
        df["diagnosis"] = LE.fit_transform(df["diagnosis"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
        st.write(df)
        st.write("Shape of dataset: ",df.shape)


        if st.button("Value Counts"):
            st.text("Value Counts By Target/Class")
            st.write(df.iloc[:,1].value_counts())
        
        # Data Visualization
        if st.button("Data Visualization"):
            st.subheader("Data Visualization")
            plt.figure(figsize=(11,5))
            st.write(sns.countplot(df['diagnosis'], palette='RdBu'))
            st.pyplot()
    
    classifier_name = st.sidebar.selectbox("Select Classifier: ",["NONE", "Logistic Regression","KNN","SVM","Decision Trees",
                                                              "Random Forest"])
    if (classifier_name == "Logistic Regression"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,log = logisticRegression(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Logistic Regression model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of Logistic Regression model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,log.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()
        
    
    elif(classifier_name == "KNN"):
        st.write("""
# Explore different models

""")
        score,report,knn = Knn_Classifier(x_train, x_test, y_train, y_test)
        st.text("Accuracy of K-Nearest Neighbour model is: ")
        st.write(score,"%")
        print('\n')
        st.text("Report of K-Nearest Neighbour model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,knn.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()

    
    elif (classifier_name == "SVM"):
        st.write("""
# Explore different classifiers
Which one is the best?
""")
        score,report,svc_linear = svm(x_train, x_test, y_train, y_test)
        st.text("Accuracy of SVM model is:")
        st.write(score,"%")
        print('\n')
        st.text("Report of SVM model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,svc_linear.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()

    elif (classifier_name == "Decision Trees"):
        st.write("""
# Explore different models
Which one is the best?
""")
        score,report,tree = Decision_Tree(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Decision Tree model is:")
        st.write(score,"%")
        print('\n')
        st.text("Report of Decision Tree model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,tree.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()


    elif (classifier_name == "Random Forest"):
        st.write("""
# Explore different modles
""")
        score,report,forest = Random_Forest(x_train, x_test, y_train, y_test)
        st.text("Accuracy of Random Forest model is:")
        st.write(score,"%")
        print('\n')
        st.text("Report of Random Forest model is: ")
        st.write(report)
        print('\n')
        cm = confusion_matrix(y_test,forest.predict(x_test))
        st.write(sns.heatmap(cm,annot=True,fmt="d", cmap="mako"))
        st.pyplot()
        



    # user input 
    choose_pred = st.sidebar.selectbox("Make a Prediction",["NONE","User Inputted Prediction"])
    
    if(choose_pred == "User Inputted Prediction"):
        
        sample_data, results =accept_user_data()
         
        st.subheader("Predict")
        if st.checkbox("Click"):
            choose_mdl = st.selectbox("Choose a Model:",["Logistic Regression","K-Nearest Neighbours","SVM", "Decision Tree", "Random Forest"])
            
            if(choose_mdl == "Logistic Regression"):
                score,report,log = logisticRegression(x_train, x_test, y_train, y_test)
                pred = log.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", LE.inverse_transform(pred))
            
            
            elif(choose_mdl == "K-Nearest Neighbours"):
                score,report,knn = Knn_Classifier(x_train, x_test, y_train, y_test)
                pred = knn.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", LE.inverse_transform(pred))
                
            elif(choose_mdl == "SVM"):
                score,report,svc_linear = svm(x_train, x_test, y_train, y_test)
                pred = svc_linear.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", LE.inverse_transform(pred))

            elif(choose_mdl == "Decision Tree"):
                score, report, tree = Decision_Tree(x_train, x_test, y_train, y_test)
                pred = tree.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", LE.inverse_transform(pred))

            elif(choose_mdl == "Random Forest"):
                score, report, forest = Random_Forest(x_train, x_test, y_train, y_test)
                pred = forest.predict(sc.transform(sample_data))
                st.write("The Predicted Class is: ", LE.inverse_transform(pred))

if __name__ == "__main__":
    main()
