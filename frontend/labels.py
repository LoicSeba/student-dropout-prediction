MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    5: "1st phase - special contingent (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE = {
    "33": "Biofuel Production Technologies",
    "171": "Animation and Multimedia Design",
    "8014": "Social Service (evening)",
    "9003": "Agronomy",
    "9070": "Communication Design",
    "9085": "Veterinary Nursing",
    "9119": "Informatics Engineering",
    "9130": "Equinculture",
    "9147": "Management",
    "9238": "Social Service",
    "9254": "Tourism",
    "9500": "Nursing",
    "9556": "Oral Hygiene",
    "9670": "Advertising and Marketing Management",
    "9773": "Journalism and Communication",
    "9853": "Basic Education",
    "9991": "Management (evening)",
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year - not completed",
    19: "Basic education 3rd cycle or equiv.",
    38: "Basic education 2nd cycle or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

NACIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldovan",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

PARENT_QUALIFICATION = {
    1: "Secondary Education - 12th Year or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle or Equiv.",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of general high school course",
    29: "9th Year - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read, no 4th year of schooling",
    37: "Basic education 1st cycle or equiv.",
    38: "Basic Education 2nd Cycle or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

PARENT_OCCUPATION = {
    0: "Student",
    1: "Legislative/Executive Directors and Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative staff",
    5: "Personal Services, Security and Sellers",
    6: "Farmers and Skilled Agricultural Workers",
    7: "Skilled Industry, Construction and Craft Workers",
    8: "Installation and Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Not specified",
}


def get_code(label_map: dict, label: str):
    """Get the raw code for a label."""
    for code, text in label_map.items():
        if text == label:
            return code
    raise ValueError(f"Unknown label: {label}")


def format_options(label_map: dict) -> list[str]:
    """Get a list of 'code — Label' selectbox options"""
    return [f"{code} — {label}" for code, label in sorted(label_map.items(), key=lambda kv: str(kv[0]))]


def code_from_option(option: str):
    """Get the code from a 'code — Label' selectbox option."""
    code_str = option.split(" — ")[0]
    return int(code_str) if code_str.lstrip("-").isdigit() else code_str


def default_index(label_map: dict, options: list[str], raw_default, key_is_str: bool = False) -> int:
    """Get the default index for a selectbox."""
    
    if raw_default is None:
        return 0
    try:
        code = str(int(float(raw_default))) if key_is_str else int(float(raw_default))
    except (TypeError, ValueError):
        return 0
 
    label = label_map.get(code)
    if label is None:
        return 0
 
    target_option = f"{code} — {label}"
    return options.index(target_option) if target_option in options else 0