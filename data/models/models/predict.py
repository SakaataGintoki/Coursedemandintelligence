import joblib
import pandas as pd

model = joblib.load("course_model.pkl")
encoder = joblib.load("encoder.pkl")

def predict(course, semester, rating, prev_enroll, difficulty):
    course_encoded = encoder.transform([course])[0]

    input_data = pd.DataFrame([{
        'course': course_encoded,
        'semester': semester,
        'faculty_rating': rating,
        'previous_enrollment': prev_enroll,
        'course_difficulty': difficulty
    }])

    prediction = model.predict(input_data)[0]
    return int(prediction)

if __name__ == "__main__":
    result = predict("AI", 5, 4.5, 80, 3)
    print("Predicted Enrollment:", result)
