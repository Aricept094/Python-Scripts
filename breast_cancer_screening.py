import json
import math
from typing import Dict, Any, List

# --- Data Tables (Simplified for WhiteAvg proxy for Middle Eastern) ---
NB_data = [0.529264]  # White
AM_data = [0.09401]   # White
AF_data = [0.218626]  # White
NR_data = [0.958303]  # White
A50NB_data = [-0.288042] # White
AFNR_data = [-0.190811] # White

AR50_data = {
    "age_lt_50": [0.578841], # White
    "age_gte_50": [0.578841] # White
}

h1_data = { # WhiteAvg Incidence Data ONLY
    "20-24": [0.0000122],
    "25-29": [0.0000741],
    "30-34": [0.0002297],
    "35-39": [0.0005649],
    "40-44": [0.0011645],
    "45-49": [0.0019525],
    "50-54": [0.0026154],
    "55-59": [0.0030279],
    "60-64": [0.0036757],
    "65-69": [0.0042029],
    "70-74": [0.0047308],
    "75-79": [0.0049425],
    "80-84": [0.0047976],
    "85-89": [0.0040106]
}

h2_data = { # WhiteAvg Competing Mortality Data ONLY
    "20-24": [0.0004412],
    "25-29": [0.0005254],
    "30-34": [0.0006746],
    "35-39": [0.0009092],
    "40-44": [0.0012534],
    "45-49": [0.001957],
    "50-54": [0.0032984],
    "55-59": [0.0054622],
    "60-64": [0.0091035],
    "65-69": [0.0141854],
    "70-74": [0.0225935],
    "75-79": [0.0361146],
    "80-84": [0.0613626],
    "85-89": [0.1420663]
}


def calculate_lifetime_risk(risk_factors: Dict[str, int], current_age: int) -> float:
    """
    Calculates the lifetime breast cancer risk percentage for Middle Eastern women
    using WhiteAvg data as a proxy.

    Args:
        risk_factors (Dict[str, int]): Dictionary of risk factor values (nbiop, agemen, age1st, nrels, hyperplasia).
        current_age (int): Patient's current age.

    Returns:
        float: Lifetime breast cancer risk percentage.
    """

    ren_beta = 0 # 0 corresponds to White data (used as proxy)

    log_rr = (risk_factors['nbiop'] * NB_data[ren_beta]) + (risk_factors['agemen'] * AM_data[ren_beta]) + \
             (risk_factors['age1st'] * AF_data[ren_beta]) + (risk_factors['nrels'] * NR_data[ren_beta]) + \
             (risk_factors['age1st'] * risk_factors['nrels'] * AFNR_data[ren_beta]) + risk_factors['hyperplasia']

    relative_risk = math.exp(log_rr)
    relative_risk_50 = math.exp(log_rr + (risk_factors['nbiop'] * A50NB_data[ren_beta]))

    t1 = current_age
    t2 = 90 # Lifetime risk up to age 90
    numbr_intvl = math.ceil(t2) - math.floor(t1)
    cumulative_h = 0
    rsk_wrk = 0

    for jj in range(1, numbr_intvl + 1):
        if numbr_intvl > 1 and 1 < jj < numbr_intvl:
            intgrl_lngth = 1
        elif numbr_intvl > 1 and jj == 1:
            intgrl_lngth = 1 - (t1 - math.floor(t1)) # Corrected line: T1(rounded down) to math.floor(t1)
        elif numbr_intvl > 1 and jj == numbr_intvl:
            intgrl_lngth = (t2 - math.floor(t2)) * (t2 > math.floor(t2)) + (t2 == math.floor(t2)) # Corrected line: t2(rounded down) to math.floor(t2)
        elif numbr_intvl == 1:
            intgrl_lngth = t2 - t1
        else:
            intgrl_lngth = 1 # Default to 1 if issues occur

        age_for_lookup = t1 + jj -1 # Calculate age for data lookup in h1 and h2

        age_group_key = None
        if 20 <= age_for_lookup <= 24: age_group_key = "20-24"
        elif 25 <= age_for_lookup <= 29: age_group_key = "25-29"
        elif 30 <= age_for_lookup <= 34: age_group_key = "30-34"
        elif 35 <= age_for_lookup <= 39: age_group_key = "35-39"
        elif 40 <= age_for_lookup <= 44: age_group_key = "40-44"
        elif 45 <= age_for_lookup <= 49: age_group_key = "45-49"
        elif 50 <= age_for_lookup <= 54: age_group_key = "50-54"
        elif 55 <= age_for_lookup <= 59: age_group_key = "55-59"
        elif 60 <= age_for_lookup <= 64: age_group_key = "60-64"
        elif 65 <= age_for_lookup <= 69: age_group_key = "65-69"
        elif 70 <= age_for_lookup <= 74: age_group_key = "70-74"
        elif 75 <= age_for_lookup <= 79: age_group_key = "75-79"
        elif 80 <= age_for_lookup <= 84: age_group_key = "80-84"
        elif 85 <= age_for_lookup <= 89: age_group_key = "85-89"
        else: continue # Skip if age out of range

        arx_rr = AR50_data["age_lt_50"][ren_beta] * relative_risk if age_for_lookup < 50 else AR50_data["age_gte_50"][ren_beta] * relative_risk

        h1 = h1_data[age_group_key][0] # Use WhiteAvg incidence data (index 0 now as only column)
        h2 = h2_data[age_group_key][0] # Use WhiteAvg competing mortality data (index 0 now as only column)

        hj = (h1 * arx_rr) + h2
        pij = ((arx_rr * h1 / hj) * math.exp(-cumulative_h)) * (1 - math.exp(-hj * intgrl_lngth))
        cumulative_h += hj * intgrl_lngth
        rsk_wrk += pij

    return 100 * rsk_wrk


def analyze_breast_cancer_screening(form_data: Dict[str, Any]) -> str:
    """
    Analyzes patient form data to determine breast cancer screening recommendations,
    including lifetime risk assessment for average-risk Middle Eastern women
    (using WhiteAvg data as proxy) and considering radiotherapy history.

    For moderate-risk patients, includes shared decision-making considerations for
    supplemental screening and *does not* provide average-risk screening recommendations.

    Args:
        form_data (Dict[str, Any]): A dictionary representing the patient's form data.

    Returns:
        str: Breast cancer screening recommendation, risk category, or error message.
    """

    # --- 1. Extract Data from Form and Check for High-Risk Factors ---
    age = form_data.get("question2")
    gender = form_data.get("question3")
    breast_biopsy_count_response = form_data.get("question19")
    abnormal_biopsy_response = form_data.get("question20") # Not used in risk calculation but could be for refinement
    age_menarche_response = form_data.get("question21")
    age_first_birth_response = form_data.get("question22")
    family_history_response = form_data.get("question23")
    screening_types = form_data.get("question1")
    radiotherapy_history_response = form_data.get("question55") # New question


    # Check if the form is filled for breast cancer screening and for women
    if not screening_types or 'Item 1' not in screening_types:
        return "Breast cancer screening was not selected. This form is for breast cancer screening recommendations."
    if gender != 'Item 1':
        return "This breast cancer screening tool is designed for women only."

    # Check for radiotherapy history (High Risk Factor)
    if radiotherapy_history_response == "Item 1": # "بله" (Yes)
        return "This breast cancer screening tool is designed for average-risk women and is not suitable for individuals with a history of chest radiotherapy between ages 10 and 30. Please consult with your physician for personalized breast cancer risk assessment and screening recommendations."
    elif radiotherapy_history_response == "Item 3": # "نمی دانم" (Don't know) -  Treat as needing clarification, but for now, proceed as if no history for average risk screening purposes.
        pass # Or you could add a message to clarify with patient, e.g., "Please clarify radiotherapy history for a more precise assessment."


    # Check for missing required questions for risk assessment
    required_questions = {
        "Age": age,
        "Breast Biopsy Count": breast_biopsy_count_response,
        "Age at Menarche": age_menarche_response,
        "Age at First Birth": age_first_birth_response,
        "Family History": family_history_response,
        "Radiotherapy History": radiotherapy_history_response  # Now required
    }
    missing_questions = [name for name, response in required_questions.items() if response is None]
    if missing_questions:
        return f"Error: Missing required information for risk assessment. Please update the form with answers for: {', '.join(missing_questions)}."


    try:
        age_int = int(age) # Use a different variable name to avoid shadowing
        if age_int < 20:
            return "Breast cancer screening guidelines are not applicable for women under 20 years old as per the provided guidelines. This tool is for women 20 years and older."
    except (ValueError, TypeError):
        return "Error: Invalid age provided. Please provide a numerical age."


    # --- 2. Process Responses for Risk Calculation ---
    # ... (Response processing code - same as before) ...
    nbiop = 0
    if breast_biopsy_count_response == "Item 2": # "یک مورد نمونه برداری"
        nbiop = 1
    elif breast_biopsy_count_response == "Item 3": # "دو مورد نمونه برداری یا بیشتر"
        nbiop = 1 # Recoded to 1

    # AgeMen (Age at menarche)
    agemen = 0
    if age_menarche_response == "Item 1": # "7 تا 11 سال"
        agemen = 1
    elif age_menarche_response == "Item 2": # "12 تا 13 سال"
        agemen = 2
    elif age_menarche_response == "Item 3": # "بیشتر از 14 سال"
        agemen = 3
    elif age_menarche_response == "Item 4": # "نمی دانم / نامشخص"
        agemen = 2 # Use average value

    # Age1st (Age at first live birth)
    age1st = 0
    if age_first_birth_response == "Item 1": # "زایمان نداشته ام"
        age1st = 3 # Nulliparous
    elif age_first_birth_response == "Item 2": # "کمتر از 20 سال"
        age1st = 0
    elif age_first_birth_response == "Item 3": # "20 تا 24 سال"
        age1st = 1
    elif age_first_birth_response == "Item 4": # "25 تا 29 سال"
        age1st = 2
    elif age_first_birth_response == "Item 5": # "30 سال یا بیشتر"
        age1st = 3
    elif age_first_birth_response == "Item 6": # "نمی دانم / نامشخص"
        age1st = 3 # Treat unknown as > 30 or nulliparous

    # NRels (Number of 1st-degree relatives with breast cancer)
    nrels = 0
    if family_history_response == "Item 2": # "یک مورد"
        nrels = 1 # Recoded to 1
    elif family_history_response == "Item 3": # "دو مورد یا بیشتر"
        nrels = 1 # Recoded to 1
    elif family_history_response == "Item 4": # "نمی دانم / نامشخص"
        nrels = 0 # Assume no family history

    hyperplasia = 0 # Assume no hyperplasia for average risk

    risk_factors = {
        "nbiop": nbiop,
        "agemen": agemen,
        "age1st": age1st,
        "nrels": nrels,
        "hyperplasia": hyperplasia
    }

    # --- 3. Calculate Lifetime Risk ---
    lifetime_risk_percentage = calculate_lifetime_risk(risk_factors, age_int)


    # --- 4. Risk Category Assessment and Screening Strategy ---
    if lifetime_risk_percentage < 15:
        risk_category = "Average Risk"
        recommendation_prefix = f"Patient is categorized as Average Risk (Lifetime risk: {lifetime_risk_percentage:.2f}%).\n"
        if age_int < 40:
            screening_recommendation = recommendation_prefix + "For women under 40 years of age at average risk, routine breast cancer screening is not recommended. Encourage breast self-awareness and reporting any breast concerns. This recommendation is for average-risk women."
        elif 40 <= age_int <= 49:
            screening_recommendation = recommendation_prefix + "For women aged 40 to 49 years at average risk, screening mammography is suggested. Screening every two years (biennial mammography) is a reasonable approach based on guidelines. Annual mammography is also an option. Shared decision-making is recommended to individualize the decision based on personal preferences and benefits vs harms of screening at different intervals. Modality: Mammography (digital mammography or digital breast tomosynthesis). This recommendation is for average-risk women."
        elif 50 <= age_int <= 74:
            screening_recommendation = recommendation_prefix + "For women aged 50 to 74 years at average risk, breast cancer screening with mammography is suggested every one to two years. Modality: Mammography (digital mammography or digital breast tomosynthesis). Frequency: Every 1-2 years. This recommendation is for average-risk women."
        elif age_int >= 75:
            screening_recommendation = recommendation_prefix + "For women aged 75 years and older at average risk, screening mammography may be offered if life expectancy is at least 10 years and after shared decision-making considering benefits and harms. If screening is elected, mammography every two years is appropriate. Modality: Mammography (digital mammography or digital breast tomosynthesis). Frequency: Every 2 years (if life expectancy >= 10 years and patient elects screening). This recommendation is for average-risk women."
        else:
            screening_recommendation = recommendation_prefix + "Could not determine screening strategy based on provided information and guidelines. This recommendation is for average-risk women."
        return screening_recommendation

    elif 15 <= lifetime_risk_percentage <= 20:
        risk_category = "Moderate Risk" # While guidelines are for average risk, we can still provide average risk recommendations but indicate caution.
        recommendation_prefix = f"Patient is categorized as Moderate Risk (Lifetime risk: {lifetime_risk_percentage:.2f}%). This breast cancer screening tool is primarily designed for average-risk women. For women at moderate risk, consider the following points and discuss supplemental screening options with your clinician:\n\n"
        supplemental_screening_info = """
        **Shared Decision-Making for Supplemental Screening (Moderate Risk):**

        *   While routine supplemental screening (ultrasound or MRI in addition to mammography) is not routinely suggested for moderate risk, it can be considered after discussion with your clinician.
        *   **Supplemental Screening Options:**
            *   **Ultrasound:** May be more accessible and less expensive than MRI.
            *   **MRI:**  Potentially higher sensitivity but may have more false positives.
        *   **Discuss with your clinician:**
            *   Your personal preferences regarding potential risks versus possible benefits of supplemental screening.
            *   Availability of supplemental screening in your area.
            *   Insurance coverage for supplemental screening (often not routinely covered for moderate risk in many US states).

        If you are interested in supplemental screening, engage in a shared decision-making discussion with your clinician to determine the best approach for you.
        \n"""
        return recommendation_prefix + supplemental_screening_info

    else:
        risk_category = "High Risk"
        return f"Based on risk assessment, the patient is categorized as {risk_category} (Lifetime risk: {lifetime_risk_percentage:.2f}%). This breast cancer screening tool is designed for average-risk women and is not suitable for individuals with higher than average risk. Please refer to guidelines for moderate-to-high risk women and consult with your physician for personalized breast cancer risk assessment and screening recommendations."



# --- Example Usage ---
if __name__ == "__main__":
    form_data_json_string_no_radio = """
    {
      "question1": [ "Item 1" ],
      "question2": "55",
      "question3": "Item 1",
      "question19": "Item 1",
      "question20": "Item 1",
      "question21": "Item 2",
      "question22": "Item 4",
      "question23": "Item 1",
      "question55": "Item 2"
    }
    """
    form_data_no_radio = json.loads(form_data_json_string_no_radio)
    recommendation_no_radio = analyze_breast_cancer_screening(form_data_no_radio)
    print("No Radiotherapy History Example (Average Risk):\n", recommendation_no_radio)

    form_data_json_string_radio_yes = """
    {
      "question1": [ "Item 1" ],
      "question2": "45",
      "question3": "Item 1",
      "question19": "Item 2",
      "question20": "Item 2",
      "question21": "Item 1",
      "question22": "Item 1",
      "question23": "Item 3",
      "question55": "Item 1"
    }
    """
    form_data_radio_yes = json.loads(form_data_json_string_radio_yes)
    recommendation_radio_yes = analyze_breast_cancer_screening(form_data_radio_yes)
    print("\nRadiotherapy History YES Example (High Risk due to Radio):\n", recommendation_radio_yes)

    form_data_json_string_radio_dk = """
    {
      "question1": [ "Item 1" ],
      "question2": "42",
      "question3": "Item 1",
      "question19": "Item 1",
      "question20": "Item 1",
      "question21": "Item 2",
      "question22": "Item 4",
      "question23": "Item 1",
      "question55": "Item 3"
    }
    """
    form_data_radio_dk = json.loads(form_data_json_string_radio_dk)
    recommendation_radio_dk = analyze_breast_cancer_screening(form_data_radio_dk)
    print("\nRadiotherapy History Don't Know Example (Average Risk Screening, but clarify Radio History):\n", recommendation_radio_dk)

    form_data_json_string_age_42 = """
    {
      "question1": [ "Item 1" ],
      "question2": "42",
      "question3": "Item 1",
      "question19": "Item 1",
      "question20": "Item 1",
      "question21": "Item 2",
      "question22": "Item 4",
      "question23": "Item 1",
      "question55": "Item 2"
    }
    """
    form_data_age_42 = json.loads(form_data_json_string_age_42)
    recommendation_age_42 = analyze_breast_cancer_screening(form_data_age_42)
    print(f"\nAge 40-49 Example (Average Risk):\n{recommendation_age_42}")

    form_data_json_string_age_35 = """
    {
      "question1": [ "Item 1" ],
      "question2": "35",
      "question3": "Item 1",
      "question19": "Item 1",
      "question20": "Item 1",
      "question21": "Item 2",
      "question22": "Item 4",
      "question23": "Item 1",
      "question55": "Item 2"
    }
    """
    form_data_age_35 = json.loads(form_data_json_string_age_35)
    recommendation_age_35 = analyze_breast_cancer_screening(form_data_age_35)
    print(f"\nAge under 40 Example (Average Risk - No Screening):\n{recommendation_age_35}")

    form_data_json_string_moderate_risk = """
    {
      "question1": [ "Item 1" ],
      "question2": "50",
      "question3": "Item 1",
      "question19": "Item 1",
      "question20": "Item 1",
      "question21": "Item 1",
      "question22": "Item 1",
      "question23": "Item 2",
      "question55": "Item 2"
    }
    """
    form_data_moderate_risk = json.loads(form_data_json_string_moderate_risk)
    recommendation_moderate_risk = analyze_breast_cancer_screening(form_data_moderate_risk)
    print(f"\nModerate Risk Example (Shared Decision-Making for Supplemental Screening):\n", recommendation_moderate_risk)

    form_data_json_string_high_risk_category = """
    {
      "question1": [ "Item 1" ],
      "question2": "45",
      "question3": "Item 1",
      "question19": "Item 2",
      "question20": "Item 2",
      "question21": "Item 1",
      "question22": "Item 1",
      "question23": "Item 3",
      "question55": "Item 2"
    }
    """
    form_data_high_risk_category = json.loads(form_data_json_string_high_risk_category)
    # Corrected line - passing the dictionary object, not the JSON string:
    recommendation_high_risk_category = analyze_breast_cancer_screening(form_data_high_risk_category)
    print(f"\nHigh Risk Category Example (Not Suitable for Average Risk Tool):\n", recommendation_high_risk_category)