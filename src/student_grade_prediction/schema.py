from __future__ import annotations

FIELD_GROUPS = [
    (
        "Student Profile",
        [
            "school",
            "sex",
            "age",
            "address",
            "famsize",
            "Pstatus",
            "guardian",
        ],
    ),
    (
        "Family Background",
        [
            "Medu",
            "Fedu",
            "Mjob",
            "Fjob",
            "reason",
            "nursery",
            "higher",
            "internet",
        ],
    ),
    (
        "Academic Record",
        [
            "traveltime",
            "studytime",
            "failures",
            "schoolsup",
            "famsup",
            "paid",
            "activities",
            "absences",
            "G1",
            "G2",
        ],
    ),
    (
        "Lifestyle and Wellbeing",
        [
            "romantic",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
        ],
    ),
]

FIELD_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Home area",
    "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing school",
    "guardian": "Primary guardian",
    "traveltime": "Travel time",
    "studytime": "Weekly study time",
    "failures": "Past class failures",
    "schoolsup": "School support",
    "famsup": "Family support",
    "paid": "Extra paid classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Plans for higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Family relationship quality",
    "freetime": "Free time after school",
    "goout": "Going out frequency",
    "Dalc": "Workday alcohol use",
    "Walc": "Weekend alcohol use",
    "health": "Current health status",
    "absences": "School absences",
    "G1": "First period grade (G1)",
    "G2": "Second period grade (G2)",
}

CATEGORY_LABELS = {
    "school": {
        "GP": "Gabriel Pereira",
        "MS": "Mousinho da Silveira",
    },
    "sex": {
        "F": "Female",
        "M": "Male",
    },
    "address": {
        "U": "Urban",
        "R": "Rural",
    },
    "famsize": {
        "LE3": "3 or fewer",
        "GT3": "More than 3",
    },
    "Pstatus": {
        "T": "Together",
        "A": "Apart",
    },
    "Mjob": {
        "teacher": "Teacher",
        "health": "Health care",
        "services": "Civil services",
        "at_home": "At home",
        "other": "Other",
    },
    "Fjob": {
        "teacher": "Teacher",
        "health": "Health care",
        "services": "Civil services",
        "at_home": "At home",
        "other": "Other",
    },
    "reason": {
        "home": "Close to home",
        "reputation": "School reputation",
        "course": "Program preference",
        "other": "Other",
    },
    "guardian": {
        "mother": "Mother",
        "father": "Father",
        "other": "Other",
    },
    "schoolsup": {
        "yes": "Yes",
        "no": "No",
    },
    "famsup": {
        "yes": "Yes",
        "no": "No",
    },
    "paid": {
        "yes": "Yes",
        "no": "No",
    },
    "activities": {
        "yes": "Yes",
        "no": "No",
    },
    "nursery": {
        "yes": "Yes",
        "no": "No",
    },
    "higher": {
        "yes": "Yes",
        "no": "No",
    },
    "internet": {
        "yes": "Yes",
        "no": "No",
    },
    "romantic": {
        "yes": "Yes",
        "no": "No",
    },
}

SLIDER_CONFIG = {
    "age": {"min": 15, "max": 22, "step": 1},
    "Medu": {"min": 0, "max": 4, "step": 1},
    "Fedu": {"min": 0, "max": 4, "step": 1},
    "traveltime": {"min": 1, "max": 4, "step": 1},
    "studytime": {"min": 1, "max": 4, "step": 1},
    "failures": {"min": 0, "max": 4, "step": 1},
    "famrel": {"min": 1, "max": 5, "step": 1},
    "freetime": {"min": 1, "max": 5, "step": 1},
    "goout": {"min": 1, "max": 5, "step": 1},
    "Dalc": {"min": 1, "max": 5, "step": 1},
    "Walc": {"min": 1, "max": 5, "step": 1},
    "health": {"min": 1, "max": 5, "step": 1},
    "G1": {"min": 0, "max": 20, "step": 1},
    "G2": {"min": 0, "max": 20, "step": 1},
}

FIELD_HELP = {
    "Medu": "0 = none, 4 = higher education",
    "Fedu": "0 = none, 4 = higher education",
    "traveltime": "1 = under 15 minutes, 4 = more than 1 hour",
    "studytime": "1 = under 2 hours, 4 = more than 10 hours",
    "famrel": "1 = very poor, 5 = excellent",
    "freetime": "1 = very low, 5 = very high",
    "goout": "1 = very low, 5 = very high",
    "Dalc": "1 = very low, 5 = very high",
    "Walc": "1 = very low, 5 = very high",
    "health": "1 = very poor, 5 = very good",
}
