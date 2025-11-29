"""
Personalized Data Science Assignment Generator
==============================================

This module generates unique assignment parameters for each student based on their student ID.
It ensures that while all students work on the same fundamental concepts, their specific
tasks are personalized to prevent direct sharing of solutions.

Usage:
    from student_assignment import generate_assignment
    assignment = generate_assignment("r0787625")
"""

import hashlib
import random
import pandas as pd

# Dataset descriptions and feature explanations
DATASET_INFO = {
    "california_housing.csv": {
        "description": (
            "California Housing Dataset - This dataset contains information about housing "
            "in California districts from the 1990 census. The goal is to predict the "
            "median house value based on various demographic and geographic features."
        ),
        "target": "median_house_value",
        "target_description": "Median house value in California districts (in hundreds of thousands of dollars)",
        "features": {
            "longitude": "Longitude coordinate of the district",
            "latitude": "Latitude coordinate of the district",
            "housing_median_age": "Median age of houses in the district",
            "total_rooms": "Total number of rooms in the district",
            "total_bedrooms": "Total number of bedrooms in the district", 
            "population": "Total population of the district",
            "households": "Total number of households in the district",
            "median_income": "Median income of households (in tens of thousands)",
            "ocean_proximity": "Proximity to ocean (categorical: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN)"
        }
    },
    "heart_disease.csv": {
        "description": (
            "Heart Disease Dataset - This dataset contains clinical measurements from patients "
            "to predict the presence of heart disease. The data includes various medical "
            "indicators and test results commonly used in cardiac diagnosis."
        ),
        "target": "target",
        "target_description": "Presence of heart disease (0 = no disease, 1-4 = disease severity levels)",
        "features": {
            "age": "Age of the patient in years",
            "sex": "Gender (1 = male, 0 = female)",
            "cp": "Chest pain type (1-4: typical angina, atypical angina, non-anginal pain, asymptomatic)",
            "trestbps": "Resting blood pressure in mm Hg",
            "chol": "Serum cholesterol level in mg/dl",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "Resting electrocardiogram results (0-2)",
            "thalach": "Maximum heart rate achieved during exercise",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of peak exercise ST segment (1-3)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia test result (3 = normal, 6 = fixed defect, 7 = reversible defect)"
        }
    }
}

def generate_assignment(student_id):
    """
    Generate a personalized assignment based on student ID.
    
    Parameters:
    -----------
    student_id : str
        The student's unique identifier (e.g., 'r0787625')
        
    Returns:
    --------
    dict : Assignment parameters including dataset, constraints, and requirements
    """
    
    # Create a deterministic seed from student ID
    seed = int(hashlib.md5(student_id.encode()).hexdigest(), 16) % (2**31)
    random.seed(seed)
    
    # Determine if this is a regression or classification task
    task_type = "regression" if seed % 2 == 0 else "classification"
    
    # Common assignment parameters
    assignment = {
        "student_id": student_id,
        "task_type": task_type,
        "seed": seed,
        "submission_requirements": {
            "format": "Jupyter Notebook (.ipynb)",
            "filename": f"DSF_Assignment_{student_id}.ipynb",
            "sections_required": [
                "0. Setup",
                "1. Data Loading and Exploration",
                "2. Data Processing", 
                "3. Model Building",
                "4. Evaluation"
            ]
        }
    }
    
    if task_type == "regression":
        assignment.update(_generate_regression_assignment(seed))
    else:
        assignment.update(_generate_classification_assignment(seed))
    
    # Add convenient task type variables for notebooks
    if 'california' in assignment['dataset']:
        assignment['TASK_TYPE'] = 'regression'
        assignment['TARGET'] = 'median_house_value'
    else:
        assignment['TASK_TYPE'] = 'classification'
        assignment['TARGET'] = 'target'
    
    return assignment

def _generate_regression_assignment(seed):
    """Generate regression-specific assignment parameters."""
    random.seed(seed)
    
    dataset_name = "california_housing.csv"
    dataset_info = DATASET_INFO[dataset_name]
    
    # California Housing Dataset parameters
    feature_subsets = [
        ["longitude", "latitude", "housing_median_age", "total_rooms"],
        ["total_bedrooms", "population", "households", "median_income"],
        ["longitude", "latitude", "median_income", "ocean_proximity"],
        ["housing_median_age", "total_rooms", "population", "median_income"],
        ["latitude", "total_bedrooms", "households", "ocean_proximity"]
    ]
    
    forbidden_features = [
        ["housing_median_age"],
        ["total_bedrooms"], 
        ["population"],
        ["households"],
        ["longitude"]
    ]
    
    required_features = random.choice(feature_subsets)
    forbidden_feature_set = random.choice(forbidden_features)
    
    models_to_try = random.sample([
        "Linear Regression",
        "Polynomial Regression", 
        "Lasso Regression",
        "Decision Trees",
        "KNN"
    ], k=3)
    
    evaluation_focus = random.choice([
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)", 
        "R² Score",
        "Combination of MAE and R²"
    ])
    
    # Get feature descriptions for assigned features
    feature_descriptions = {feat: dataset_info["features"][feat] for feat in required_features}
    
    return {
        "dataset": dataset_name,
        "dataset_description": dataset_info["description"],
        "target_variable": dataset_info["target"],
        "target_description": dataset_info["target_description"],
        "task_description": "Predict median house values in California districts\nUse different regression models to compare performance.",
        "required_features": required_features,
        "feature_descriptions": feature_descriptions,
        "forbidden_features": forbidden_feature_set,
        "models_to_compare": models_to_try,
        "primary_evaluation_metric": evaluation_focus,
        "min_visualizations": 5,
        "special_requirements": _get_special_requirements(seed, "regression")
    }

def _generate_classification_assignment(seed):
    """Generate classification-specific assignment parameters.""" 
    random.seed(seed)
    
    dataset_name = "heart_disease.csv"
    dataset_info = DATASET_INFO[dataset_name]
    
    # Heart Disease Dataset parameters
    feature_subsets = [
        ["age", "sex", "cp", "trestbps"],
        ["chol", "fbs", "restecg", "thalach"], 
        ["exang", "oldpeak", "slope", "ca"],
        ["age", "cp", "thalach", "oldpeak"],
        ["sex", "chol", "exang", "thal"]
    ]
    
    forbidden_features = [
        ["fbs"],
        ["restecg"],
        ["slope"], 
        ["ca"],
        ["thal"]
    ]
    
    required_features = random.choice(feature_subsets)
    forbidden_feature_set = random.choice(forbidden_features)
    
    # For classification, we have exactly 3 models, so we'll give each student 3 models
    # but in random order to maintain some variation
    models_list = ["Logistic Regression", "Decision Trees", "KNN"]
    random.shuffle(models_list)
    models_to_try = models_list
    
    evaluation_focus = random.choice([
        "Accuracy and Precision",
        "Recall and F1-Score",
        "ROC-AUC and Precision-Recall AUC",
        "Confusion Matrix Analysis"
    ])
    
    # Get feature descriptions for assigned features
    feature_descriptions = {feat: dataset_info["features"][feat] for feat in required_features}
    
    return {
        "dataset": dataset_name,
        "dataset_description": dataset_info["description"],
        "target_variable": dataset_info["target"],
        "target_description": dataset_info["target_description"],
        "task_description": "Predict presence of heart disease based on clinical measurements.\nUse various classification models to compare performance.",
        "required_features": required_features,
        "feature_descriptions": feature_descriptions,
        "forbidden_features": forbidden_feature_set,
        "models_to_compare": models_to_try,
        "primary_evaluation_metric": evaluation_focus,
        "min_visualizations": 4,
        "special_requirements": _get_special_requirements(seed, "classification")
    }

def _get_special_requirements(seed, task_type):
    """Generate additional personalized requirements."""
    random.seed(seed)
    
    base_requirements = [
        "Create at least one custom feature through feature engineering",
        "Handle missing values with justification for your approach",
        "Perform cross-validation for model evaluation"
    ]
    
    additional_options = [
        "Include feature importance analysis",
        "Perform outlier detection and handling",
        "Create interaction features between variables", 
        "Apply feature scaling/normalization",
        "Implement early stopping in applicable models",
        "Compare performance across different train/test splits"
    ]
    
    # Each student gets 1-2 additional requirements
    additional_reqs = random.sample(additional_options, k=random.randint(1, 2))
    
    return base_requirements + additional_reqs


def print_assignment_details(student_id):
    """
    Print comprehensive assignment details for a given student ID.
    This is a clean wrapper function for notebook use.
    """
    assignment = generate_assignment(student_id)
    
    print(f"Assignment for {student_id}:")
    print(f"Dataset: {assignment['dataset']}")
    
    print(f"\nDataset Description:")
    print(f"{assignment['dataset_description']}")
    
    print(f"\nTarget: {assignment['target_variable']} - {assignment['target_description']}")
    
    print(f"\nRequired features:")
    for feature in assignment['required_features']:
        print(f"  - {feature}: {assignment['feature_descriptions'][feature]}")

    print(f"\nTask type:\n-{assignment['task_type']}")
    print(f"Task description:\n-{assignment['task_description']}")
    # Return assignment parameters for notebook use
    return assignment

# Example usage and testing
if __name__ == "__main__":
    # Test with example student IDs
    test_students = ["r0787625", "s1234567", "u9876543"]
    
    for student_id in test_students:
        assignment = generate_assignment(student_id)
        print_assignment_details(student_id)
        print("\n" + "="*60 + "\n")