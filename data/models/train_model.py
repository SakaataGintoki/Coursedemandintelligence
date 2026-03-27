import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("../data/course_data.csv")

# Encode course
le = LabelEncoder()
data['course'] = le.fit_transform(data['course'])

X = data.drop("expected_enrollment", axis=1)
y = data["expected_enrollment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))

joblib.dump(model, "course_model.pkl")
joblib.dump(le, "encoder.pkl")

print("Model saved!")
