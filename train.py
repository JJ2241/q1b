from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Training started..")

def train_model():
    X, y = load_iris(return_X_y=True)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier().fit(Xtrain, ytrain)
    print(f"Model Accuracy: {model.score(Xtest, ytest)}")

    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    train_model()