from typing import Dict







SHEET_NOMINAL = "תשואה נומינלית הנגזרת מהמודל"
SHEET_REAL    = "תשואה ריאלית הנגזרת מהמודל"
SHEET_GAP     = "התפתחות פער התשואות"
SCENARIO_COL  = "Unnamed: 2"            # העמודה שמכילה מזהה תרחיש ('מדדי'/'קלנדרי')
SCENARIO = None                 # מדדי או קלנדרי
HORIZON       = 360                      # חודשים


# ---------- קונפיג ברירת מחדל ----------
DURATIONS_MONTHS_DEFAULT = range(12, 361, 1) #range(12, 372, 12)

# ------------------------------ HEBREW MONTHS ------------------------------
HE_MONTHS = {
    'ינואר': 1, 'פברואר': 2, 'מרץ': 3, 'אפריל': 4, 'מאי': 5, 'יוני': 6,
    'יולי': 7, 'אוגוסט': 8, 'ספטמבר': 9, 'אוקטובר': 10, 'נובמבר': 11, 'דצמבר': 12
}

ANCHOR_TRACK_MAP = {
    "מלצ": "Variable Non-Indexed",
    "מצ":  "Variable Indexed",
    "פריים": "Prime",
    "קלצ": "Fixed Non-Indexed",
    "קצ":  "Fixed Indexed",
    "מקמ": "Makam",
    'מטח יורו': 'matah dolar',
    'מטח דולר': 'matah euro',
}

bank_of_israel_rate = 4.0
prime_margin = 1.5
defult_capital_allocation = "75%"

# מינימום ומקסימום על מסלולים קבועים ומשתנים
MIN_fixed_share = 0.33 
MAX_variable_share = 0.66
sensitivity="בינוני"
prepay_window_key="לא"
objective_mode="balanced"
alpha=0.5
monthly_income_factor= 3.5 # monthly_income_net=float(first_payment * 3.5),

# מינימום הכנסה ביחס להחזר המשכנתא
OFFICIAL_ratio_limit = 0.50
INTERNAL_ratio_limit = 0.4

banks_options = {
    "": None,
    "בנק הפועלים": "hapoalim",
    "בנק לאומי": "leomi",
    "בנק מזרחי": "mizrahi",
    "בנק דיסקונט": "disconut"
}
# קל"צ: {שנים: ריבית באחוזים}
FIXED_NON_INDEXED_TABLE: Dict[int, float] = {4: 4.4,
 5: 4.4,
 6: 4.4,
 7: 4.4,
 8: 4.5,
 9: 4.5,
 10: 4.5,
 11: 4.5,
 12: 4.5,
 13: 4.5,
 14: 4.5,
 15: 4.6,
 16: 4.6,
 17: 4.6,
 18: 4.6,
 19: 4.6,
 20: 4.6,
 21: 4.6,
 22: 4.6,
 23: 4.7,
 24: 4.7,
 25: 4.7,
 26: 4.7,
 27: 4.7,
 28: 4.7,
 29: 4.7,
 30: 4.7}

# ק"צ: {שנים: ריבית באחוזים}
FIXED_INDEXED_TABLE: Dict[int, float] = {4: 2.8,
 5: 2.8,
 6: 2.85,
 7: 2.85,
 8: 2.85,
 9: 2.85,
 10: 2.9,
 11: 2.9,
 12: 2.9,
 13: 2.9,
 14: 2.9,
 15: 2.9,
 16: 2.95,
 17: 2.95,
 18: 2.95,
 19: 3.0,
 20: 3.0,
 21: 3.0,
 22: 3.0,
 23: 3.0,
 24: 3.0,
 25: 3.0,
 26: 3.0,
 27: 3.0,
 28: 3.0,
 29: 3.0,
 30: 3.0}

SPREADS: Dict[str, Dict[str, float]] = {'Variable Indexed': 1.1, 'Variable Non-Indexed': 0.9, 'Prime': -0.7}


# התאמות לפי הקצאת הון (LTV) פר סוג מסלול (אם לא קיים ערך – 0.0)
LOAN_ADJ_RULES: Dict[str, Dict[str, float]] = {'Fixed Non-Indexed': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5},
 'Fixed Indexed': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5},
 'Variable Non-Indexed': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5},
 'Variable Indexed': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5},
 'Prime': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5},
 'Makam': {'100%': 2.0, '35%': 0.0, '50%': 0.0, '60%': 0.1, '75%': 0.1, 'any': 1.5}}

# ---------- סיווג מסלולים ----------
FIXED_TYPES = {"קלצ", "קצ"}  # קבועה לא צמודה/צמודה
VARIABLE_TYPES = {"מלצ", "מצ", "פריים", "מקמ", "מטח דולר", "מטח יורו"}
PRIME_TYPES = {"פריים"}

# מיפוי למסלולי ה-InterestRateCalculator
ANCHOR_TRACK_MAP = {
    "מלצ": "Variable Non-Indexed",
    "מצ": "Variable Indexed",
    "פריים": "Prime",
    "קלצ": "Fixed Non-Indexed",
    "קצ": "Fixed Indexed",
    "מקמ" : "Makam",
}


# optimization general difinitions
MIN_ACTIVE_SHARE = 0.1      # למשל 5% מההלוואה לכל מסלול "דולק"
MAX_SHARE_PER_OPTION = 0.4

TIME_LIMIT = 10 # sec
no_savings = 20000.0 # no_savings money between ori and anouther mortage
diff_between_opt = 150000.0
max_diff_of_first_payment = 500
max_diff_of_max_payment = 500


manual_interest_increase = 0.0

# ------------------------------ DISCOUNT PARAMS ------------------------------
# הנחת רגישות (באחוזים)
SENSITIVITY_DISCOUNT = {'נמוך': 0.0, 'בינוני': 25.0, 'גבוה': 40.0}

# הנחת פירעון מוקדם (באחוזים)
PREPAY_DISCOUNT = {'כן': {'משתנה': {1: 25.0, 12: 25.0, 24: 25.0, 30: 25.0, 36: 25.0, 60: 0.0, 84: 0.0, 120: 0.0},
        'קבועה': -25.0},
 'לא': {'משתנה': {1: 0.0, 12: 0.0, 24: 0.0, 30: 0.0, 36: 0.0, 60: 0.0, 84: 0.0, 120: 0.0},
        'קבועה': 0.0},
 'לא בטוח': {'משתנה': {1: 12.5, 12: 12.5, 24: 12.5, 30: 12.5, 36: 12.5, 60: 0.0, 84: 0.0, 120: 0.0},
             'קבועה': -12.5}}


def durations_months(max_months):
    return list(range(12, max_months, 1))

