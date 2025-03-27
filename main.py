import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def dataclean():
    data = pd.read_csv("data.csv")
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def create_models(data):
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
        trained_models[name] = model
    
    return trained_models, scalar

def start():
    data = dataclean()
    models, scalar = create_models(data)
    
    with open("model/scalar.pkl", 'wb') as f:
        pickle.dump(scalar, f)
    
    for name, model in models.items():
        with open(f"model/{name.replace(' ', '_').lower()}.pkl", 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    start()
