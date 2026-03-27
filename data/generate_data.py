import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        'course': np.random.choice(['AI', 'Data Science', 'Cybersecurity', 'Cloud', 'Business Analytics'], n),
        'semester': np.random.randint(1, 9, n),
        'faculty_rating': np.random.uniform(3.0, 5.0, n),
        'previous_enrollment': np.random.randint(20, 120, n),
        'course_difficulty': np.random.randint(1, 5, n)
    })

    data['expected_enrollment'] = (
        data['previous_enrollment'] * 0.6 +
        data['faculty_rating'] * 10 -
        data['course_difficulty'] * 5 +
        np.random.randint(-10, 10, n)
    ).astype(int)

    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("course_data.csv", index=False)
    print("Data generated successfully!")
