# pipeline.py

from src.data_generation import generate_course_data
from src.train import train_subject
import pandas as pd

def run_pipeline():
    print("🚀 Starting ML Pipeline...")

    # Step 1: Data Generation
    df = generate_course_data(n_samples=2000)
    print("✅ Data generated")

    # Step 2: Train models for all subjects
    subjects = df['course_name'].unique()
    summary = []

    for subject in subjects:
        print(f"\n📊 Training for: {subject}")
        results, best_name, best_model, subdf, _ = train_subject(subject, df, plot=False)

        if results is not None:
            summary.append({
                "Subject": subject,
                "Best Model": best_name,
                "R2": results.loc[best_name]["R² Score"]
            })

    summary_df = pd.DataFrame(summary)
    print("\n📋 Pipeline Completed")
    print(summary_df)

    return summary_df


if __name__ == "__main__":
    run_pipeline() 
