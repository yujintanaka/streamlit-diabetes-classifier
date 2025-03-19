import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle as pickle


def create_model(data):

    # over sampling of the dataset to get a balanced dataset
    class_0 = data[data['Diabetes_binary'] == 0]
    class_1 = data[data['Diabetes_binary'] == 1]

    # over sampling of the minority class 1
    class_1_over = class_1.sample(len(class_0), replace=True)

    # Creating a new dataframe with over sampled class 1 df and class 0 df
    df_over = pd.concat([class_1_over, class_0], axis=0)

    X = df_over.drop(['Diabetes_binary'], axis=1)
    y = df_over['Diabetes_binary']

    # Scale only the continuous features using column transformer
    # refer to Jupyter Notebook for finding these features
    scaler = ColumnTransformer([
        ('scaler', StandardScaler(), 
        [
            'BMI',
            'GenHlth',
            'MentHlth',
            'PhysHlth',
            'Age',
            'Education',
            'Income'
        ])
    ], remainder='passthrough')
    X = scaler.fit_transform(X)


    # train test split
    X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.2, random_state=42)

    # training
    rfc = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', min_samples_split=10, random_state=0)
    rfc.fit(X_train, y_train)

    # testing
    y_pred = rfc.predict(X_test)
    print('Accuracy of Model: ', accuracy_score(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test,y_pred))    

    return rfc, scaler


def main():
    data = pd.read_csv('data/train_data.csv')
    small_data = pd.read_csv('data/data.csv')
    small_df = small_data.sample(frac=0.01, random_state=42)
    model, scaler = create_model(data)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)
    with open ('data/small_df.csv', 'wb') as f:
        small_df.to_csv(f, index=False)



if __name__ == "__main__":
    main()