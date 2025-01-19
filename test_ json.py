import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class PatientData:
    question2: int  # Age
    question4: Optional[str] = None  # Current smoker?
    question28: Optional[str] = None  # Former smoker?
    question30: Optional[float] = None  # How many years did you smoke?
    question29: Optional[float] = None  # How many years since you quit smoking?

def is_indicated_for_lung_cancer_screening(patient_data: PatientData) -> bool:

    age = patient_data.question2

    currently_smokes = patient_data.question4 == "Item 2"
    previously_smoked = patient_data.question28 == "Item 2"

    years_smoked = patient_data.question30 or 0.0
    years_since_quit = patient_data.question29 or 0.0

    if 50 <= age <= 80:
        if currently_smokes or (previously_smoked and years_since_quit <= 15.0):
            pack_years = years_smoked  # Assuming 1 pack per day
            if pack_years >= 20.0:
                return True

    return False

def read_json_from_file(path: str) -> PatientData:

    with open(path, 'r') as f:
        data = json.load(f)

    return PatientData(
        question2=data['question2'],
        question4=data.get('question4'),
        question28=data.get('question28'),
        question30=data.get('question30'),
        question29=data.get('question29'),
    )

if __name__ == "__main__":
    file_path = "/home/aricept094/mydata/question_data.json"  

    try:
        patient_data = read_json_from_file(file_path)
        if is_indicated_for_lung_cancer_screening(patient_data):
            print("Patient is indicated for lung cancer screening.")
        else:
            print("Patient is not indicated for lung cancer screening.")
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {file_path}")
    except KeyError as e:
        print(f"Error: Missing field in JSON data: {e}")
    except TypeError as e:
        print(f"Error: Incorrect data type in JSON: {e}")