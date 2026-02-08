
# =============================================================================
# 🧮  IMPORTS AND GLOBAL SETTINGS
# =============================================================================


import config as con
from datetime import date, datetime
from dataclasses import dataclass
import pulp as pl
import numpy as np
from typing import  List,Dict, Callable, Tuple, Optional
import re
import mortgage_data_loader as dl
import pandas as pd 
import json
import os 

# --- Initialize Data Locally (No dependency on data_loader_service) ---
# This uses the V2 loader which caches the request via st.cache_resource
XLSX_PATH = dl.fetch_latest_boi_excels()
print(f'extracted latest_boi_excels from: {XLSX_PATH}')
DATASTORE = dl.load_workbook_data(XLSX_PATH, con.HORIZON, con.SCENARIO)

ancors = dl.load_boi_data()
NOMINAL_ANCHOR = ancors["nominal_anchor"]
REAL_ANCHOR = ancors["real_anchor"]
MAKAM_ANCOR = ancors["makam_anchor"]


# =============================================================================
# CLASS: INTEREST RATE CALCULATOR
# =============================================================================

class InterestRateCalculator:
    """
    מחלקה לחישוב ריביות מותאמות לכל מסלול — קבוע, משתנה, פריים ומק"מ.
    """
    def __init__(self, bank_of_israel_rate: float = con.bank_of_israel_rate, prime_margin: float = con.prime_margin,
                 fixed_non_indexed_table: Optional[Dict[int, float]] = None,
                 fixed_indexed_table: Optional[Dict[int, float]] = None,
                 spreads: Optional[Dict[str, Dict[str, float]]] = None,
                 ):
        """
        טוען עוגנים *למסלולים משתנים* מהקבצים הכי חדשים בתיקיית boi_yields,
        ומאפשר הזנה ידנית (או Injection בפרמטרים) למסלולים הקבועים ולחוקי הקצאת הון.
        """
        self.bank_rate = float(bank_of_israel_rate)
        self.prime_rate = float(bank_of_israel_rate + prime_margin)

        self.Anchor = {
            "Variable Non-Indexed": NOMINAL_ANCHOR,  # שנים→ריבית (אחוזים)
            "Variable Indexed":     REAL_ANCHOR,     # שנים→ריבית (אחוזים)
        }

        # --- קבוע: שולף ממילון ידני (ניתן override בפרמטרים) ---
        self.Fixed_Non_Indexed = dict(con.FIXED_NON_INDEXED_TABLE)
        self.Fixed_Indexed     = dict(con.FIXED_INDEXED_TABLE)
        if fixed_non_indexed_table:
            self.Fixed_Non_Indexed.update(fixed_non_indexed_table)
        if fixed_indexed_table:
            self.Fixed_Indexed.update(fixed_indexed_table)

        # --- מרווחים וחוקי הקצאת הון ---
        self.Spread = dict(con.SPREADS)
        if spreads:
            for k, v in spreads.items():
                self.Spread.setdefault(k, {}).update(v)

        self.Rules_loan_percentage = dict(con.LOAN_ADJ_RULES)
        self.makam_ancor = MAKAM_ANCOR
    @staticmethod
    def _bucket_val(d: Dict[str, float], bucket: str) -> float:
        return float(d.get(bucket, d.get("any", 0.0))) if d else 0.0

    @staticmethod
    def _nearest_year_key(years_key: int, table: Dict[int, float]) -> Optional[int]:
        """מציאת שנת מפתח קרובה בטבלת ריביות."""
        if not table:
            return None
        if years_key in table:
            return years_key
        keys = sorted(table.keys())
        for k in keys:
            if k >= years_key:
                return k
        return keys[-1] if keys else None

    def get_adjusted_rate(
        self,
        track_type: str,
        loan_percentage: str,
        change_freq_months: int,
        remain_months: int
    ) -> Optional[float]:
        """חישוב ריבית מתואמת למסלול בהתאם לעוגן, מרווח והקצאת הון."""
        loan_adj_global = float(self.Rules_loan_percentage[track_type].get(loan_percentage, 0.0)) 
        
        if "Prime" in track_type:
            
            years_key = int(np.ceil((remain_months) / 12)) 
            base = self.prime_rate 
            spr  = self.Spread["Prime"] # self._bucket_val(self.Spread.get("Prime", {}), years_key) # -מרווח -0.8
            sum_rate = float(base + spr + loan_adj_global)
            print(f'>>> get_adjusted_rate | track_type:{track_type}, loan_percentage: {loan_percentage}, base:{base}, spr:{spr}, loan_adj_global:{loan_adj_global}, sum_rate:{sum_rate}')
            return sum_rate

        if track_type.startswith("Variable"):
            
            if not change_freq_months:
                raise ValueError("Variable track requires change_freq_months.")
            years_key = int(np.ceil(change_freq_months / 12))
            anchor_dict = self.Anchor["Variable Non-Indexed"] if "Non-Indexed" in track_type else self.Anchor["Variable Indexed"]
            spr = self.Spread["Variable Non-Indexed"] if "Non-Indexed" in track_type else self.Spread["Variable Indexed"]
            anchor_rate = anchor_dict.get(years_key)
            if anchor_rate is None:
                return None
            sum_rate = float(anchor_rate + spr + loan_adj_global)
            print(f'>>> get_adjusted_rate | track_type:{track_type}, loan_percentage: {loan_percentage},change_freq_months:{change_freq_months}, anchor_rate:{anchor_rate}, spr:{spr}, loan_adj_global:{loan_adj_global}, sum_rate:{sum_rate}')

            return sum_rate

        if track_type == "Fixed Non-Indexed" or track_type == "Fixed Indexed":
            
            years_key = int(np.ceil((remain_months) / 12))  
            table = self.Fixed_Non_Indexed if track_type == "Fixed Non-Indexed" else self.Fixed_Indexed
            yk = self._nearest_year_key(years_key, table)
            if yk is None:
               
                raise ValueError(f"לא נמצאה ריבית קבועה עבור {track_type} לשנים {years_key}. מלאי את הטבלה הידנית.")
            base_rate = table.get(yk)

            sum_rate = float(base_rate + loan_adj_global)
            print(f'>>> get_adjusted_rate | track_type:{track_type}, loan_percentage: {loan_percentage}, years_key:{years_key}, base:{base_rate}, loan_adj_global:{loan_adj_global}, sum_rate:{sum_rate}')

            return float(base_rate + loan_adj_global)

        if track_type == 'Makam':
            base_rate = self.makam_ancor
            spr = self.Spread["Makam"]
            return float(base_rate + spr + loan_adj_global)
        
        
        
        return 0

# =============================================================================
#  GENERAL HELPERS
# =============================================================================

def monthly_to_yearly(monthly_rate: float) -> float:
    """המרת ריבית חודשית לשנתית."""
    return (1 + monthly_rate) ** 12 - 1

def _interp_zero_yield_at_tenor(z_curve: Dict[int, float], tenor_m: int) -> float:
    """אינטרפולציה של ריבית אפס לפי טנור."""
    if tenor_m in z_curve:
        return float(z_curve[tenor_m])
    keys = sorted(z_curve.keys())
    if tenor_m <= keys[0]:
        return float(z_curve[keys[0]])
    if tenor_m >= keys[-1]:
        return float(z_curve[keys[-1]])
    for i in range(1, len(keys)):
        if keys[i] >= tenor_m:
            lo, hi = keys[i - 1], keys[i]
            y0, y1 = z_curve[lo], z_curve[hi]
            w = (tenor_m - lo) / (hi - lo)
            return float(y0 + w * (y1 - y0))
    return float(list(z_curve.values())[-1])

def build_anchor_path_from_zero_curve(z_curve: Dict[int, float], V: int, term_months: int) -> List[float]:
    """בניית מסלול עוגן פשוט לפי עקום ריביות-אפס."""
    if V <= 0:
        raise ValueError("V must be positive (months)")
    if term_months <= 0:
        return []

    anchor: List[float] = [0.0] * term_months
    A_V = _interp_zero_yield_at_tenor(z_curve, V)
    first_window_len = min(V, term_months)
    for t in range(first_window_len):
        anchor[t] = A_V

    k = 2
    while (k - 1) * V < term_months:
        start = (k - 1) * V + 1
        end = min(k * V, term_months)
        A_prev = _interp_zero_yield_at_tenor(z_curve, (k - 1) * V)
        A_curr = _interp_zero_yield_at_tenor(z_curve, k * V)
        gross = (1.0 + A_curr) ** (k * V / 12.0) / (1.0 + A_prev) ** (((k - 1) * V) / 12.0)
        f_annual = gross ** (12.0 / V) - 1.0
        for t in range(start - 1, end):
            anchor[t] = f_annual
        k += 1

    return anchor

def prepare_prime_monthly_rates(months: int, boi_annual_percent: List[float], margin_percent: float) -> List[float]:
    """בניית מסלול ריביות חודשי לפריים."""
    if len(boi_annual_percent) < months:
        last = boi_annual_percent[-1]
        boi_annual_percent = boi_annual_percent + [last] * (months - len(boi_annual_percent))
    else:
        boi_annual_percent = boi_annual_percent[:months]
    return [((boi + margin_percent) / 12.0 / 100.0) for boi in boi_annual_percent]

# =============================================================================
# INTEREST TYPE DETECTION & LOAN TRACK PREPARATION
# =============================================================================
def create_interest_type(loan_tracks):
    """
    גוזר טיפוס ריבית מילוני loan_tracks:
    - Prime
    - Fixed Indexed / Fixed Non-Indexed
    - Variable Indexed / Variable Non-Indexed
    """
    
    for track, rec in loan_tracks.items():

        rec['current_interest_type'] = rec['Interest_Type']
        # לוג-בקרה
        # print('track:', track, ' current_interest_type:', current)
    return loan_tracks

def _as_date(x):
    """הבטחת ערך כתאריך."""
    return x.date() if hasattr(x, "date") else x

def _months_between(d1, d2):
    """חישוב מספר חודשים בין שני תאריכים."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def compute_months_to_next_reset(rec: dict) -> Optional[int]:

    """
    מחזיר כמה חודשים נשארו עד תאריך האיפוס הבא (אם יש).
    אם התאריך כבר עבר, מזיזים במחזורים של freq עד שנקבל תאריך עתידי.
    """
    try:
        ctype = rec.get('current_interest_type') or rec.get('current_intreset_type')
        val = rec.get('Interest_Rate_Change_Frequency_in_Months')

        if pd.isna(val):
            freq = 0

        else:
            freq = int(val)

        #freq  = int(rec.get('Interest_Rate_Change_Frequency_in_Months') or 0)
        next_dt = rec.get('Next_Interest_Rate_Change_Date')
        today  = rec.get('latter_date_object')
    except:
        pass
    # רק למסלולים משתנים/פריים עם תדירות > 0 יש איפוסים
    
    is_variable = ctype in (
        "Prime", "פריים",
        "Variable Indexed Interest", "Variable Non-Indexed Interest",
        "מצ", "מלצ", "מקמ"
    )
    if not (is_variable and freq > 0 and next_dt and today):
        return None

    next_dt = _as_date(next_dt)
    today   = _as_date(today)

    # אם התאריך הבא עבר, לקדם במחזורים של freq עד שנגיע לעתיד
    # (למקרה שהמסמך ישן או שהמערכת רצה אחרי האיפוס)
    months_ahead = _months_between(today, next_dt)
    if months_ahead < 0:
        # כמה מחזורים צריך להוסיף?
        cycles = (-months_ahead + freq - 1) // freq
        # מקדמים תאריך קדימה cycles*freq חודשים
        y, m = next_dt.year, next_dt.month + cycles * freq
        y += (m - 1) // 12
        m  = ((m - 1) % 12) + 1
        next_dt = date(y, m, min(next_dt.day, 28))  # זהירות עם סוף-חודש
        months_ahead = _months_between(today, next_dt)

    return max(0, months_ahead)

def create_total_months(loan_tracks):
    """יצירת שדות עזר לזמן כולל ולזמן שנותר."""
    for track, rec in loan_tracks.items():
        last_payment_date   = _as_date(rec['Expected_Last_Payment_Date'])
        first_payment_date  = _as_date(rec['First_Payment_Date'])
        latter_date_object  = _as_date(rec['latter_date_object'])

        rec['total_months_Remained'] = max(0, _months_between(latter_date_object, last_payment_date))
        rec['total_months_passed']   = max(0, _months_between(first_payment_date, latter_date_object))
        rec['total_months']          = max(0, _months_between(first_payment_date, last_payment_date))

        # חדש: חודשים עד האיפוס הבא (אם רלוונטי)
        rec['months_to_next_reset'] = compute_months_to_next_reset(rec)

    return loan_tracks

# =============================================================================
# ADAPTIVE ANCHOR AND RATE PATH BUILDERS
# =============================================================================

def build_anchor_path_from_zero_curve_adaptive_rachel(z_curve: Dict[int, float], V: int, term_months: int,months_to_next_reset:int) -> List[float]:
    if V <= 0:
        raise ValueError("V must be positive (months)")
    if term_months <= 0:
        return []
    #import pdb;pdb.set_trace()
    anchor: List[float] = [0.0] * term_months
    A_V = _interp_zero_yield_at_tenor(z_curve, V)
    first_window_len = min(min(V, months_to_next_reset),term_months)
    for t in range(first_window_len):
        anchor[t] = A_V

    k = 2
    offset_month = V - months_to_next_reset
    #
    try:
        while (k - 1) * V < term_months:
            start = (k - 1) * V + 1 - offset_month
            end = min(k * V - offset_month, term_months)
            A_prev = _interp_zero_yield_at_tenor(z_curve, ((k - 1) * V - offset_month))
            A_curr = _interp_zero_yield_at_tenor(z_curve, (k * V - offset_month))
            gross = (1.0 + A_curr) ** ((k * V  - offset_month)/ 12.0) / (1.0 + A_prev) ** (((k - 1) * V   - offset_month) / 12.0)
            f_annual = gross ** (12.0 / (V - offset_month)) - 1.0
            for t in range(start - 1, end):
                anchor[t] = f_annual
            k += 1
    except:
        pass 
    return anchor

def build_monthly_rates_adaptive(
    rate_annual_start: float,
    zero_series: List[float],
    freq: int,
    months: int,
    months_to_next_reset: int = 0
) -> List[float]:
    """
    ממיר את נתיב העוגן (אחוזים שנתיים) לריביות חודשיות בפועל סביב rate_annual_start.
    """
    z_curve_nominal = {m: r for m, r in enumerate(zero_series, start=1)}
    
    if months_to_next_reset == 0:
        anchor_path = build_anchor_path_from_zero_curve(z_curve_nominal, V=freq, term_months=months)
    else:
        anchor_path = build_anchor_path_from_zero_curve_adaptive_rachel(z_curve_nominal, V=freq, term_months=months, months_to_next_reset=months_to_next_reset)
        
    
    anchor_annual_pct = [a * 100.0 for a in anchor_path]  # -> אחוז
    base_anchor = anchor_annual_pct[0]
    anchor_annual_pp = [v - base_anchor for v in anchor_annual_pct]  # סטיות עוגן מנורמלות לנק' פתיחה
    return [((rate_annual_start + dp) / 12.0 / 100.0) for dp in anchor_annual_pp]

def calculate_spitzer_adaptive(
    loan,
    months,
    Periodic_inflation,
    Change_compared_to_the_base_index,
    monthly_rates,
    rate_type,
    freq_rate_change,
    months_to_next_reset: int = 0,
    round_agorot: bool = False,
):
    """
    אם המסלול משתנה/פריים:
      - האיפוס הראשון יתרחש לאחר months_to_next_reset חודשים,
        ואז כל freq_rate_change חודשים.
    אם months_to_next_reset==0 => איפוס מיידי בחודש הראשון (כלומר לוח חדש).
    במסלולים קבועים/ללא freq: אין איפוסים.
    """
    def _pmt(rate, nper, pv):
        if nper <= 0:  return pv
        if abs(rate) < 1e-12: return pv / nper
        return rate * pv / (1.0 - (1.0 + rate) ** (-nper))

    def _normalize_rates_monthly(rates):
        rs = list(map(float, rates))
        if max(abs(x) for x in rs) > 0.02:
            return [x / 100.0 for x in rs]
        return rs

    def _cum_K_from_cpi(cpi_decimals):
        K, k = [], 1.0
        for v in cpi_decimals:
            k *= (1.0 + v)
            K.append(k)
        return K

    def _normalize_cpi_and_K(cpi_series, K_given):
        cpi_series = list(map(float, cpi_series))
        cpi1 = cpi_series;  K1 = _cum_K_from_cpi(cpi1)
        cpi2 = [x / 100.0 for x in cpi_series];  K2 = _cum_K_from_cpi(cpi2)
        if K_given and len(K_given) >= 1:
            def mre(Ka, Kb):
                n = min(len(Ka), len(Kb))
                if n == 0: return float("inf")
                err = 0.0
                for i in range(n):
                    if Kb[i] == 0: continue
                    err += abs(Ka[i] - Kb[i]) / max(1e-12, abs(Kb[i]))
                return err / n
            return (cpi1, K1) if mre(K1, K_given) <= mre(K2, K_given) else (cpi2, K2)
        avg = sum(abs(x) for x in cpi_series) / max(1, len(cpi_series))
        return (cpi2, K2) if avg > 0.02 else (cpi1, K1)

    if months <= 0:
        raise ValueError("months must be positive")
    if loan <= 0:
        raise ValueError("loan must be positive")
    if len(Periodic_inflation) < months:
        raise ValueError("Periodic_inflation length is shorter than months")
    if len(monthly_rates) < months:
        raise ValueError("monthly_rates length is shorter than months")

    r_monthly = _normalize_rates_monthly(monthly_rates)
    K_given = (list(map(float, Change_compared_to_the_base_index[:months]))
               if Change_compared_to_the_base_index is not None and len(Change_compared_to_the_base_index) >= months else None)
    cpi_decimals, _ = _normalize_cpi_and_K(Periodic_inflation[:months], K_given)

    schedule = []
    Opening_balance = float(loan)
    K_prev = 1.0
    prev_payment = None
    a_real_current = None

    # חישוב “היסט משמרת” לאיפוס הראשון:
    # אם המסלול משתנה/פריים ויש תדירות, נתרגם months_to_next_reset ל-offset
    # כך ש־reset יקרה כאשר (m-1 + offset) % freq == 0
    is_variable = rate_type in ("פריים", 'מצ', 'מלצ', "מקמ", "Variable Indexed Interest", "Variable Non-Indexed Interest", "Prime")
    freq = int(freq_rate_change or 0)
    offset = 0
    #import pdb; 
    if is_variable and freq > 0:
        offset = (freq - (months_to_next_reset or 0)) % freq  # אם next=0 → offset=0 → איפוס מיידי
    #print(f'offset is: {offset}, freq: {freq}')
    for m in range(1, months + 1):
        cpi_m = cpi_decimals[m - 1]
        K_t   = K_prev * (1.0 + cpi_m)
        balance_indexed = Opening_balance * (1.0 + cpi_m)
        r_m = float(r_monthly[m - 1])

        # כלל האיפוס:
        needs_reset = (m == 1)  # תמיד בחודש ראשון נחשב PMT מתאים
        if is_variable and freq > 0:
            # reset במחזורים לפי offset
            if ((m - 1 + offset) % freq) == 0:
                needs_reset = True
        #print(f'm:{m}, needs_reset:{needs_reset}')
        if needs_reset:
            n_remaining  = months - m + 1
            real_balance = balance_indexed / max(1e-12, K_t)
            a_real_current = _pmt(r_m, n_remaining, real_balance)
            current_payment = a_real_current * K_t
        else:
            current_payment = prev_payment * (1.0 + cpi_m)

        interest_payment  = balance_indexed * r_m
        max_needed = balance_indexed + interest_payment
        if current_payment > max_needed:
            current_payment = max_needed
            if a_real_current is not None:
                a_real_current = current_payment / max(1e-12, K_t)

        principal_payment = current_payment - interest_payment
        principal_payment_nominal = principal_payment / max(1e-12, K_t)
        indexation_payment = principal_payment - principal_payment_nominal
        Closeing_balance = balance_indexed - principal_payment
        if Closeing_balance < 1e-8:
            Closeing_balance = 0.0

        if round_agorot:
            current_payment = round(current_payment, 2)
            interest_payment = round(interest_payment, 2)
            principal_payment = round(principal_payment, 2)
            principal_payment_nominal = round(principal_payment_nominal, 2)
            indexation_payment = round(indexation_payment, 2)
            Closeing_balance = round(Closeing_balance, 2)

        used_rate = (1.0 + r_m) * (1.0 + cpi_m) - 1.0
        used_rate_yearly = ((1.0 + used_rate) ** 12 - 1.0) * 100.0
        

        schedule.append([
            m,
            Opening_balance,
            current_payment,
            interest_payment,
            principal_payment,
            principal_payment_nominal,
            indexation_payment,
            Closeing_balance,
            used_rate_yearly,
        ])

        Opening_balance = Closeing_balance
        K_prev = K_t
        prev_payment = current_payment

        if Opening_balance <= 0.0:
            break

    return schedule

def calculate_keren_shava_adaptive(
    loan,
    months,
    Periodic_inflation,
    Change_compared_to_the_base_index,
    monthly_rates,
):
    def _normalize_rates_monthly(rates):
        rates = list(map(float, rates))
        if max(abs(x) for x in rates[:months]) > 0.2:
            return [x / 100.0 for x in rates]
        return rates

    def _cum_K_from_cpi(cpi_decimals):
        K = []; k = 1.0
        for v in cpi_decimals:
            k *= (1.0 + v); K.append(k)
        return K

    def _normalize_cpi_and_K(cpi_series, K_given):
        cpi_series = list(map(float, cpi_series))
        cpi1 = cpi_series; K1 = _cum_K_from_cpi(cpi1)
        cpi2 = [x / 100.0 for x in cpi_series]; K2 = _cum_K_from_cpi(cpi2)
        if K_given and len(K_given) >= 1:
            def mre(Ka, Kb):
                n = min(len(Ka), len(Kb))
                if n == 0: return float("inf")
                err = 0.0
                for i in range(n):
                    if Kb[i] == 0: continue
                    err += abs(Ka[i] - Kb[i]) / max(1e-12, abs(Kb[i]))
                return err / n
            return (cpi1, K1) if mre(K1, K_given) <= mre(K2, K_given) else (cpi2, K2)
        avg = sum(abs(x) for x in cpi_series) / max(1, len(cpi_series))
        return (cpi2, K2) if avg > 0.02 else (cpi1, K1)

    if months <= 0: raise ValueError("months must be positive")
    if loan <= 0: raise ValueError("loan must be positive")
    if len(Periodic_inflation) < months: raise ValueError("Periodic_inflation length is shorter than months")
    if len(monthly_rates) < months: raise ValueError("monthly_rates length is shorter than months")

    r_monthly = _normalize_rates_monthly(monthly_rates[:months])
    K_given = (list(map(float, Change_compared_to_the_base_index[:months]))
               if Change_compared_to_the_base_index is not None and len(Change_compared_to_the_base_index) >= months else None)
    cpi_decimals, K_series = _normalize_cpi_and_K(Periodic_inflation[:months], K_given)

    # בקרן שווה – רכיב הקרן (בבסיס ריאלי) קבוע; איפוסים לא משנים אותו, רק את הריבית r_m
    real_principal_const = float(loan) / float(months)

    schedule = []
    Opening_balance = float(loan)
    K_prev = 1.0

    for m in range(1, months + 1):
        cpi_m = cpi_decimals[m - 1]
        K_t   = K_prev * (1.0 + cpi_m)
        balance_indexed = Opening_balance * (1.0 + cpi_m)
        r_m = float(r_monthly[m - 1])

        principal_payment_nominal_indexed = min(real_principal_const * K_t, balance_indexed)
        interest_payment = balance_indexed * r_m
        current_payment  = interest_payment + principal_payment_nominal_indexed
        principal_payment_nominal_base = principal_payment_nominal_indexed / K_t if K_t != 0 else principal_payment_nominal_indexed
        indexation_payment = principal_payment_nominal_indexed - principal_payment_nominal_base
        Closing_balance = balance_indexed - principal_payment_nominal_indexed
        if Closing_balance < 1e-8: Closing_balance = 0.0

        used_rate = (1.0 + r_m) * (1.0 + cpi_m) - 1.0
        used_rate_yearly = ((1.0 + used_rate) ** 12 - 1.0) * 100.0

        schedule.append([
            m,
            Opening_balance,
            current_payment,
            interest_payment,
            principal_payment_nominal_indexed,
            principal_payment_nominal_base,
            indexation_payment,
            Closing_balance,
            used_rate_yearly,
        ])

        Opening_balance = Closing_balance
        K_prev = K_t
        if Opening_balance <= 0.0: break

    return schedule

def calculate_balloon_full_adaptive(
    loan,
    months,
    Periodic_inflation,
    Change_compared_to_the_base_index,
    monthly_rates,
    round_agorot: bool = False,
    emit_final_cashflow_row: bool = True,
):
    def _normalize_rates_monthly(rates):
        rates = list(map(float, rates))
        if len(rates) and max(abs(x) for x in rates[:min(len(rates), months)]) > 0.2:
            return [x / 100.0 for x in rates]
        return rates

    def _cum_K_from_cpi(cpi_decimals):
        K, k = [], 1.0
        for v in cpi_decimals:
            k *= (1.0 + v); K.append(k)
        return K

    def _normalize_cpi_and_K(cpi_series, K_given):
        cpi_series = list(map(float, cpi_series))
        cpi1 = cpi_series;  K1 = _cum_K_from_cpi(cpi1)
        cpi2 = [x / 100.0 for x in cpi_series];  K2 = _cum_K_from_cpi(cpi2)
        if K_given and len(K_given) >= 1:
            def mre(Ka, Kb):
                n = min(len(Ka), len(Kb))
                if n == 0: return float("inf")
                err = 0.0
                for i in range(n):
                    if Kb[i] == 0: continue
                    err += abs(Ka[i] - Kb[i]) / max(1e-12, abs(Kb[i]))
                return err / n
            return (cpi1, K1) if mre(K1, K_given) <= mre(K2, K_given) else (cpi2, K2)
        avg = sum(abs(x) for x in cpi_series) / max(1, len(cpi_series))
        return (cpi2, K2) if avg > 0.02 else (cpi1, K1)

    if months <= 0: raise ValueError("months must be positive")
    if loan <= 0: raise ValueError("loan must be positive")
    if len(Periodic_inflation) < months: raise ValueError("Periodic_inflation length is shorter than months")
    if len(monthly_rates) < months: raise ValueError("monthly_rates length is shorter than months")

    r_monthly = _normalize_rates_monthly(monthly_rates[:months])
    K_given = (list(map(float, Change_compared_to_the_base_index[:months]))
               if Change_compared_to_the_base_index is not None and len(Change_compared_to_the_base_index) >= months else None)
    cpi_decimals, _ = _normalize_cpi_and_K(Periodic_inflation[:months], K_given)

    schedule = []
    Opening_balance = float(loan)
    K_prev = 1.0
    last_used_rate_yearly = 0.0

    for m in range(1, months + 1):
        cpi_m = float(cpi_decimals[m - 1])
        K_t   = K_prev * (1.0 + cpi_m)
        balance_indexed = Opening_balance * (1.0 + cpi_m)
        display_opening = balance_indexed
        r_m = float(r_monthly[m - 1])
        interest_accrued = balance_indexed * r_m

        current_payment = 0.0
        interest_payment_display = 0.0
        principal_payment_indexed = 0.0
        principal_payment_base    = 0.0
        indexation_payment        = 0.0

        Closing_balance = balance_indexed + interest_accrued
        used_rate = (1.0 + r_m) * (1.0 + cpi_m) - 1.0
        last_used_rate_yearly = ((1.0 + used_rate) ** 12 - 1.0) * 100.0

        if round_agorot:
            current_payment = round(current_payment, 2)
            interest_payment_display = round(interest_payment_display, 2)
            principal_payment_indexed = round(principal_payment_indexed, 2)
            principal_payment_base    = round(principal_payment_base, 2)
            indexation_payment        = round(indexation_payment, 2)
            Closing_balance           = round(Closing_balance, 2)
            display_opening           = round(display_opening, 2)

        schedule.append([
            m,
            display_opening,
            current_payment,
            interest_payment_display,
            principal_payment_indexed,
            principal_payment_base,
            indexation_payment,
            Closing_balance,
            last_used_rate_yearly,
        ])

        Opening_balance = Closing_balance
        K_prev = K_t

    if emit_final_cashflow_row:
        end_balance = Opening_balance
        K_T = K_prev
        principal_indexed = loan * K_T
        principal_base    = loan
        indexation_pay    = principal_indexed - principal_base
        interest_total    = end_balance - principal_indexed
        current_payment   = end_balance
        close_after       = 0.0

        if round_agorot:
            principal_indexed = round(principal_indexed, 2)
            principal_base    = round(principal_base, 2)
            indexation_pay    = round(indexation_pay, 2)
            interest_total    = round(interest_total, 2)
            current_payment   = round(current_payment, 2)

        schedule.append([
            months + 1,
            end_balance,
            current_payment,
            interest_total,
            principal_indexed,
            principal_base,
            indexation_pay,
            close_after,
            last_used_rate_yearly,
        ])

    return schedule

def calculate_balloon_partial_adaptive(
    loan,
    months,
    Periodic_inflation,
    Change_compared_to_the_base_index,
    monthly_rates,
    round_agorot: bool = False,
    emit_final_cashflow_row: bool = True,
):
    def _normalize_rates_monthly(rates):
        rates = list(map(float, rates))
        if len(rates) and max(abs(x) for x in rates[:min(len(rates), months)]) > 0.2:
            return [x / 100.0 for x in rates]
        return rates

    def _cum_K_from_cpi(cpi_decimals):
        K, k = [], 1.0
        for v in cpi_decimals:
            k *= (1.0 + v); K.append(k)
        return K

    def _normalize_cpi_and_K(cpi_series, K_given):
        cpi_series = list(map(float, cpi_series))
        cpi1 = cpi_series;  K1 = _cum_K_from_cpi(cpi1)
        cpi2 = [x / 100.0 for x in cpi_series];  K2 = _cum_K_from_cpi(cpi2)
        if K_given and len(K_given) >= 1:
            def mre(Ka, Kb):
                n = min(len(Ka), len(Kb))
                if n == 0: return float("inf")
                err = 0.0
                for i in range(n):
                    if Kb[i] == 0: continue
                    err += abs(Ka[ i] - Kb[i]) / max(1e-12, abs(Kb[i]))
                return err / n
            return (cpi1, K1) if mre(K1, K_given) <= mre(K2, K_given) else (cpi2, K2)
        avg = sum(abs(x) for x in cpi_series) / max(1, len(cpi_series))
        return (cpi2, K2) if avg > 0.02 else (cpi1, K1)

    if months <= 0: raise ValueError("months must be positive")
    if loan <= 0: raise ValueError("loan must be positive")
    if len(Periodic_inflation) < months: raise ValueError("Periodic_inflation length is shorter than months")
    if len(monthly_rates) < months: raise ValueError("monthly_rates length is shorter than months")

    r_monthly = _normalize_rates_monthly(monthly_rates[:months])
    K_given = (list(map(float, Change_compared_to_the_base_index[:months]))
               if Change_compared_to_the_base_index is not None and len(Change_compared_to_the_base_index) >= months else None)
    cpi_decimals, _ = _normalize_cpi_and_K(Periodic_inflation[:months], K_given)

    schedule = []
    Opening_balance = float(loan)
    K_prev = 1.0
    last_used_rate_yearly = 0.0

    for m in range(1, months + 1):
        cpi_m = float(cpi_decimals[m - 1])
        K_t   = K_prev * (1.0 + cpi_m)
        balance_indexed = Opening_balance * (1.0 + cpi_m)
        display_opening = balance_indexed
        r_m = float(r_monthly[m - 1])

        # תשלום ריבית בלבד (interest-only)
        interest_payment = balance_indexed * r_m
        current_payment = interest_payment
        principal_payment_indexed = 0.0
        principal_payment_base    = 0.0
        indexation_payment        = 0.0
        Closing_balance = balance_indexed  # לא יורד עד הבלון

        used_rate = (1.0 + r_m) * (1.0 + cpi_m) - 1.0
        last_used_rate_yearly = ((1.0 + used_rate) ** 12 - 1.0) * 100.0

        if round_agorot:
            current_payment = round(current_payment, 2)
            interest_payment = round(interest_payment, 2)
            Closing_balance = round(Closing_balance, 2)
            display_opening = round(display_opening, 2)

        schedule.append([
            m,
            display_opening,
            current_payment,
            interest_payment,
            principal_payment_indexed,
            principal_payment_base,
            indexation_payment,
            Closing_balance,
            last_used_rate_yearly,
        ])

        Opening_balance = Closing_balance
        K_prev = K_t

    if emit_final_cashflow_row:
        end_balance = Opening_balance
        K_T = K_prev
        principal_indexed = loan * K_T
        principal_base    = loan
        indexation_pay    = principal_indexed - principal_base
        interest_row      = 0.0
        current_payment   = principal_indexed
        close_after       = 0.0

        if round_agorot:
            principal_indexed = round(principal_indexed, 2)
            principal_base    = round(principal_base, 2)
            indexation_pay    = round(indexation_pay, 2)
            current_payment   = round(current_payment, 2)

        schedule.append([
            months + 1,
            end_balance,
            current_payment,
            interest_row,
            principal_indexed,
            principal_base,
            indexation_pay,
            close_after,
            last_used_rate_yearly,
        ])

    return schedule

def summarize_schedule(schedule: List[List[float]]):
    total_principal = 0.0
    total_interest  = 0.0
    total_index     = 0.0
    for _,_,_,interest_payment,_,principal_payment_nominal,indexation_payment,_,_ in schedule:
        total_principal += principal_payment_nominal
        total_interest  += interest_payment
        total_index     += indexation_payment
    return total_principal, total_interest, total_index

def aggregate_yearly(schedule: List[List[float]]):
    last_month = schedule[-1][0]
    years = (last_month - 1) // 12 + 1
    pr_y  = [0.0] * years
    in_y  = [0.0] * years
    idx_y = [0.0] * years
    for month,_,_,interest_payment,_,principal_payment_nominal,indexation_payment,_,_ in schedule:
        y = (month - 1) // 12
        pr_y[y]  += principal_payment_nominal
        in_y[y]  += interest_payment
        idx_y[y] += indexation_payment
    return pr_y, in_y, idx_y

def get_inflation_series():
    return DATASTORE.infl_monthly, DATASTORE.infl_K

def get_zero_series(horizon: int, sheet_name: str) -> List[float]:
    series = DATASTORE.zero_nominal if sheet_name == con.SHEET_NOMINAL else DATASTORE.zero_real
    if len(series) >= horizon:
        return series[:horizon].tolist()
    else:
        return (np.concatenate([series, np.full(horizon - len(series), series[-1])])).tolist()

# ------------------------------ ANCHOR VIS ------------------------------

def _interp_zero(z: dict, t_m: int) -> float:
    if t_m in z:   return float(z[t_m])
    keys = sorted(z.keys())
    if t_m <= keys[0]: return float(z[keys[0]])
    if t_m >= keys[-1]: return float(z[keys[-1]])
    for i in range(1, len(keys)):
        if keys[i] >= t_m:
            lo, hi = keys[i - 1], keys[i]
            y0, y1 = z[lo], z[hi]
            w = (t_m - lo) / (hi - lo)
            return y0 + w * (y1 - y0)
    return float(z[keys[-1]])

def build_anchor(z_series: list, V: int, term_m: int) -> list:
    z = {m: r for m, r in enumerate(z_series, start=1)}
    A_V = _interp_zero(z, V)
    out = [0.0] * term_m
    for t in range(min(V, term_m)):
        out[t] = A_V
    k = 2
    while (k - 1) * V < term_m:
        prev = _interp_zero(z, (k - 1) * V)
        curr = _interp_zero(z, k * V)
        gross = (1 + curr) ** (k * V / 12) / (1 + prev) ** (((k - 1) * V) / 12)
        f = (gross ** (12 / V)) - 1
        for t in range((k - 1) * V, min(k * V, term_m)):
            out[t] = f
        k += 1
    return out

# ------------------------------ DISPATCHER ------------------------------
def calculate_schedule(principal, months, annual_rate, schedule_type, rate_type, freq_rate_change, prime_margin: float = con.prime_margin,prime_offset = 0,months_to_next_reset=0):
    #print('calculate_schedule:',principal, months, annual_rate, schedule_type, rate_type, freq_rate_change,prime_margin,months_to_next_reset)
    #import pdb;pdb.set_trace()
    if rate_type == "קלצ":
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months
        monthly_rates = [(annual_rate) / 12 / 100] * months

    elif rate_type == "קצ":
        Periodic_inflation, Change_compared_to_the_base_index = get_inflation_series()
        monthly_rates = [(annual_rate) / 12 / 100] * months

    elif rate_type == "מלצ":
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months
        zero_series = get_zero_series(con.HORIZON, con.SHEET_NOMINAL)
        rate = annual_rate
        monthly_rates = build_monthly_rates_adaptive(rate, zero_series, freq_rate_change, months,months_to_next_reset)

    elif rate_type == "מצ":
        Periodic_inflation, Change_compared_to_the_base_index = get_inflation_series()
        zero_series = get_zero_series(con.HORIZON, con.SHEET_REAL)
        rate = annual_rate
        monthly_rates = build_monthly_rates_adaptive(rate, zero_series, freq_rate_change, months,months_to_next_reset)
        #monthly_rates = build_monthly_rates(rate, zero_series, freq_rate_change, months)

    elif rate_type == "פריים":
        boi_series_percent = DATASTORE.boi_forward_annual_pct
        rate_to_add = prime_margin+prime_offset
        #import pdb;pdb.set_trace()
        monthly_rates = prepare_prime_monthly_rates(months, boi_series_percent, rate_to_add)
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months

    elif rate_type == "מטח דולר":
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months
        zero_series = get_zero_series(con.HORIZON, con.SHEET_NOMINAL)
        rate = annual_rate
        monthly_rates = build_monthly_rates_adaptive(rate, zero_series, freq_rate_change, months,months_to_next_reset)

    elif rate_type == "מטח יורו":
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months
        zero_series = get_zero_series(con.HORIZON, con.SHEET_NOMINAL)
        rate = annual_rate
        monthly_rates = build_monthly_rates_adaptive(rate, zero_series, freq_rate_change, months,months_to_next_reset)

    elif rate_type == "מקמ":
        Periodic_inflation = [0.0] * months
        Change_compared_to_the_base_index = [1.0] * months
        zero_series = get_zero_series(con.HORIZON, con.SHEET_NOMINAL)
        freq_rate_change = 12
        rate = annual_rate
        monthly_rates = build_monthly_rates_adaptive(rate, zero_series, freq_rate_change, months,months_to_next_reset)

    elif rate_type == "מענק":
        schedule = []
        for m in range(months):
            schedule.append([m, 0, 0, 0, 0, 0, 0, 0, 0])
        return schedule

    elif rate_type == "זכאות":
        Periodic_inflation, Change_compared_to_the_base_index = get_inflation_series()
        rate = annual_rate
        monthly_rates = [rate / 12 / 100] * months

    else:
        
        raise ValueError(f"Error in identification rate_type={rate_type}")

    if schedule_type.startswith("שפיצר") or schedule_type.startswith("רציפש"):
        return calculate_spitzer_adaptive(principal, months, Periodic_inflation, Change_compared_to_the_base_index, monthly_rates, rate_type, freq_rate_change,months_to_next_reset)
        #return calculate_spitzer(principal, months, Periodic_inflation, Change_compared_to_the_base_index, monthly_rates, rate_type, freq_rate_change)
    if schedule_type.startswith("קרן"):
        return calculate_keren_shava_adaptive(principal, months, Periodic_inflation, Change_compared_to_the_base_index, monthly_rates)
    if schedule_type == "בלון מלא":
        return calculate_balloon_full_adaptive(
            principal, months,
            Periodic_inflation, Change_compared_to_the_base_index,
            monthly_rates,
        )
    if schedule_type == "בלון חלקי":
        return calculate_balloon_partial_adaptive(
            principal, months,
            Periodic_inflation, Change_compared_to_the_base_index,
            monthly_rates,
        )
    if schedule_type == "כפי יכולתך":
        raise NotImplementedError("מסלול 'כפי יכולתך' טרם ממומש.")
    return None

def _is_fixed(t: str) -> bool: return t in con.FIXED_TYPES
def _is_variable(t: str) -> bool: return t in con.VARIABLE_TYPES
def _is_cpi(t: str) -> bool: return t in {"קצ", "מצ"}

def _map_rate_type_to_anchor_track(rate_type: str) -> Optional[str]:
    return con.ANCHOR_TRACK_MAP.get(rate_type)

# ---------- טיפוס מועמד ----------
@dataclass
class CandidateOption:
    key: str
    rate_type: str
    schedule_type: str
    months: int
    annual_rate: float
    freq_rate_change: int
    cost_per_unit: float
    pmt1_per_unit: float
    cost_per_unit_norm:float
    pmt1_per_unit_norm:float
    eff: float
    eff_dis: float
    max_payment_in_all_mortage: float
    original_principal: float

# ---------- עזרי קלט ----------
def _safe_int(x, default=0):
    try:
        if x is None: return default
        if isinstance(x, str):
            xs = x.strip()
            if xs == "": return default
            return int(float(xs))
        if isinstance(x, (int, float)): return int(x)
        return default
    except Exception:
        return default

# ---------- ספק ריביות מהעוגנים ----------
def _make_anchor_rate_provider(bank_of_israel_rate: float,
                               prime_margin: float,
                               loan_percentage_bucket: str):
    
    calc = InterestRateCalculator(
        bank_of_israel_rate=bank_of_israel_rate,
        prime_margin=prime_margin)
        
    
    def _provider(rate_type_he: str, change_freq_months: int, remain_months: int):
        track = _map_rate_type_to_anchor_track(rate_type_he)
        if track is None:
            return None
        try:
            return calc.get_adjusted_rate(
                track_type=track,
                loan_percentage=loan_percentage_bucket, # "35%"/"50%"/...
                change_freq_months=int(change_freq_months or 0),
                remain_months=int(remain_months or 0),
            )
        except Exception:
            return None
    return _provider

# ---------- קטלוג פנימי של מסלולים – מורחב להתאמה לטבלה ----------
def _default_catalog() -> List[Dict]:
    catalog = []
    # קבועות
    catalog += [
        {"rate_type": "קלצ", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": 0},
        {"rate_type": "קצ", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": 0},
    ]
    # משתנה לא צמודה – תדירויות נפוצות (אך מעונש בקוד, לא ייבחר)
    for years in [ 2, 2.5, 3, 5, 7, 10]:
        catalog.append({"rate_type": "מלצ", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": int(years * 12)})
    # משתנה צמודה – תדירויות נפוצות, עם דגש על 24 ו-60 להתאמה לטבלה
    for years in [ 2, 2.5, 3, 5, 7, 10]:
        catalog.append({"rate_type": "מצ", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": int(years * 12)})
    # פריים (מעונש, לא ייבחר)
    catalog.append({"rate_type": "פריים", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": 1})
    # מק"מ (אופציונלי)
    catalog.append({"rate_type": "מקמ", "schedule_t": "שפיצר (החזר חודשי קבוע)", "freq": 12})
    return catalog

# ---------- יצירת מועמדים  ----------
def generate_candidate_mixtures(
    calculate_schedule: Callable,
    summarize_schedule: Callable,
    durations_months: Optional[List[int]] = None,
    rate_provider: Optional[Callable[[str, int, int], Optional[float]]] = None,routes_data_ori = None,
    *,
    ref_principal: float,
    loan_pct_bucket: str
) -> List[CandidateOption]:
    if durations_months is None:
        durations_months = con.DURATIONS_MONTHS_DEFAULT
    options: List[CandidateOption] = []
    for tr in _default_catalog():
        rate_type_t = tr["rate_type"]
        schedule_t = tr["schedule_t"]
        freq = _safe_int(tr["freq"], 60)
        for m in durations_months:
            m_int = _safe_int(m, 0)
            if not (0 < m_int <= 360):
                continue
            eff_rate = None
            if rate_provider is not None:
                eff_rate = rate_provider(rate_type_t, freq, m_int)
            if eff_rate is None:
                continue
            try:
                #import pdb;pdb.set_trace()
                kwargs = dict(
                    principal=float(ref_principal),
                    months=int(m_int),
                    annual_rate=float(eff_rate),
                    #indexation=_indexation_for_rate(rate_type),
                    schedule_type=schedule_t,
                    rate_type=rate_type_t,
                    freq_rate_change=int(freq),
                )
                
                if rate_type_t == "פריים":
                    prime_offset = con.LOAN_ADJ_RULES["Prime"][loan_pct_bucket] + con.SPREADS['Prime']
                    kwargs["prime_margin"] = con.prime_margin   
                    kwargs["prime_offset"] = prime_offset
                
                sch = calculate_schedule(**kwargs)
                max_payment_in_all_mortage = max([i[2] for i in sch])
                pr, it, idx = summarize_schedule(sch)
                total_cost = pr + it + idx
                cost_per_unit = float(total_cost) #/ float(ref_principal)
                pmt1 = float(sch[0][2]) if sch and len(sch[0]) >= 3 else 0.0
                pmt1_per_unit = pmt1 #/ float(ref_principal)
                #import pdb;pdb.set_trace()

            except Exception:
                import pdb;pdb.set_trace()
                continue

            key = f"{rate_type_t}_{m_int}_f{freq}"
            options.append(CandidateOption(
                key=key,
                rate_type=rate_type_t,
                schedule_type=schedule_t,
                months=int(m_int),
                annual_rate=float(eff_rate),
                freq_rate_change=int(freq),
                cost_per_unit=float(cost_per_unit),
                pmt1_per_unit=float(pmt1_per_unit),
                cost_per_unit_norm = 0.0,
                pmt1_per_unit_norm =  0.0,
                eff =  0.0,
                eff_dis =  0.0,
                max_payment_in_all_mortage=max_payment_in_all_mortage,
                original_principal = 0.0
            ))
    #print(f' >>> >>> routes_data_ori:{routes_data_ori}')
    if routes_data_ori:
        print(f'len(routes_data_ori) = {len(routes_data_ori)}')
        for tr_ori in routes_data_ori:
            rate_type_t = tr_ori[0]['מסלול']
            m_int = tr_ori[0]['תקופה (חודשים)']
            schedule_t = tr_ori[0]['סוג מסלול']
            eff_rate = tr_ori[0]['ריבית']
            freq = tr_ori[0]['תדירות שינוי']
            
            if not freq:
                freq = 0 
            
            relevant_prime_offset = eff_rate - (con.prime_margin + con.bank_of_israel_rate)
            print(f'rate_type_t:{rate_type_t},m_int:{m_int},schedule_t:{schedule_t},eff_rate:{eff_rate},')
            kwargs = dict(principal=float(ref_principal), months=int(m_int),annual_rate=float(eff_rate),schedule_type=schedule_t,rate_type=rate_type_t,freq_rate_change=int(freq),prime_margin=con.prime_margin,prime_offset=relevant_prime_offset)
            sch = calculate_schedule(**kwargs)
            max_payment_in_all_mortage = max([i[2] for i in sch])
            pr, it, idx = summarize_schedule(sch)
            total_cost = pr + it + idx
            cost_per_unit = float(total_cost) #/ float(ref_principal)
            pmt1 = float(sch[0][2]) if sch and len(sch[0]) >= 3 else 0.0
            pmt1_per_unit = pmt1 #/ float(ref_principal)
            original_principal = tr_ori[0]['סכום']
            key = f"ORI_{rate_type_t}_{m_int}_f{freq}"
            options.append(CandidateOption(
                    key=key,
                    rate_type=rate_type_t,
                    schedule_type=schedule_t,
                    months=int(m_int),
                    annual_rate=float(eff_rate),
                    freq_rate_change=int(freq),
                    cost_per_unit=float(cost_per_unit),
                    pmt1_per_unit=float(pmt1_per_unit),
                    cost_per_unit_norm = 0.0,
                    pmt1_per_unit_norm =  0.0,
                    eff =  0.0,
                    eff_dis =  0.0,
                    max_payment_in_all_mortage=max_payment_in_all_mortage,
                    original_principal=original_principal
                ))

        
    #import pdb;pdb.set_trace() 
    #with open("/Users/user/Desktop/BI/debug_file.txt", "w", encoding="utf-8") as f:
    #    for opti in options:
    #        f.write(str(opti))
    #        f.write('\n')
    #    f.close()
    #import pdb;pdb.set_trace()
    return options

# רגולציה על התמהיל
def check_constraints(
    *,
    loan_amount: float,
    ltv_input: str,
    monthly_income_net: float,
    sensitivity: str,
    max_monthly_payment: float,
) -> Tuple[bool, List[str], Dict[str, float]]:
    errs: List[str] = []

    if loan_amount <= 0: errs.append("סכום הלוואה חייב להיות חיובי.")
    if monthly_income_net <= 0: errs.append("הכנסה חודשית נטו חייבת להיות חיובית.")
    
    if monthly_income_net*con.INTERNAL_ratio_limit <  max_monthly_payment:
        errs.append("החזר חודשי גבוה מידי עבור ההכנסה החודשית הקיימת")
    print(f"monthly_income_net:{monthly_income_net}, max_monthly_payment:{max_monthly_payment} ")
    
    min_fixed_share = con.MIN_fixed_share
    max_variable_share = con.MAX_variable_share

    derived = {
        "ltv_limit": ltv_input,
        "min_fixed_share": float(min_fixed_share),
        "max_variable_share": float(max_variable_share),
        "official_ratio_limit": con.OFFICIAL_ratio_limit,
        "internal_ratio_limit": con.INTERNAL_ratio_limit,
    }
    return (len(errs) == 0), errs, derived

def minmax_with_floor(values: dict[str, float], floor: float = 0.30) -> dict[str, float]:
    # floor ∈ (0,1). floor=0.30 → המינימום ימופה ל-0.30 במקום 0
    
    xs = np.array(list(values.values()), dtype=float)
    vmin, vmax = float(xs.min()), float(xs.max())
    rng = max(vmax - vmin, 1e-12)  # הגנה מנumerics
    return {k: floor + (1.0 - floor) * ((values[k] - vmin) / rng) for k in values}

# ---------- פותר LP ----------
def _solve_lp(
    options: List[CandidateOption],
    loan_amount: float,
    min_fixed_share: float,
    max_variable_share: float,
    prepay_window_key: str,
    *,
    objective_mode: str = "balanced",
    alpha: float = 0.5,
    sensitivity: str = "בינוני",
    max_monthly_payment: float = 6000,
) -> Optional[Dict[str, float]]:
    #import pdb;pdb.set_trace()

    if not options or pl is None:
        return None

    # סינון ראשוני לפי יכולת החזר
    # הוסר הסינון האגרסיבי שבודק pmt1_per_unit * 0.4
    # כדי לאפשר חריגות רגעיות או מסלולים יקרים בכמויות קטנות.
    # המגבלה הכללית על ההחזר החודשי (בשורות 1390) עדיין קיימת ומחייבת את *סה"כ* התמהיל.
    
    if not options:
        return None

    # === הגדרת בעיית LP ===
    prob = pl.LpProblem("MortgageMixOptimization", pl.LpMinimize)

    # משתנים
    x = {o.key: pl.LpVariable(f"x_{o.key}", lowBound=0) for o in options}
    y = {o.key: pl.LpVariable(f"y_{o.key}", cat="Binary") for o in options}

    BIG_M = loan_amount
    EPS = con.MIN_ACTIVE_SHARE * loan_amount

    # === עלויות מנורמלות ===
    base = {o.key: float(o.cost_per_unit) for o in options}
    base_norm = minmax_with_floor(base, floor=0.30)

    pmt1 = {o.key: float(o.pmt1_per_unit) for o in options}
    pmt1_norm = minmax_with_floor(pmt1, floor=0.30)

    # sensitivity_discount and prepay_discount moved to config.py
    # Using con.SENSITIVITY_DISCOUNT and con.PREPAY_DISCOUNT

    eff_cost = {}

    for o in options:
        # פונקציית מטרה base
        if objective_mode == "total_cost":
            eff = base_norm[o.key]
        elif objective_mode == "pmt1":
            eff = pmt1_norm[o.key]
        else:
            eff = alpha * base_norm[o.key] + (1 - alpha) * pmt1_norm[o.key]

        
        if _is_fixed(o.rate_type):
            discount = con.SENSITIVITY_DISCOUNT[sensitivity] / 100.0 + con.PREPAY_DISCOUNT[prepay_window_key]["קבועה"] / 100.0
        else:
            # Here we assume the config structure is flat for variable (just one value) 
            # OR we maintain the frequency keys. 
            # In config I wrote: "כן": {"משתנה": 25, ...} which is a single int.
            # But the original code relied on frequency: prepay_discount["כן"]["משתנה"][o.freq_rate_change]
            # Wait, my config update simplified it to a single value. 
            # I must check if I broke the logic.
            # Original: {1:25, 12:25...}. Effectively all relevant freqs got 25.
            # So a single value is likely fine, optimization-wise.
            try:
                # Try dict access if I kept it deep? No, I wrote int in config.
                var_discount_val = con.PREPAY_DISCOUNT[prepay_window_key]["משתנה"]
                if isinstance(var_discount_val, dict):
                     discount = var_discount_val.get(o.freq_rate_change, 0) / 100.0
                else:
                     discount = var_discount_val / 100.0
            except:
                discount = 0.0

        eff_cost[o.key] = eff * (1 - discount)

    # === חלוקת מסלולים ל-ORI ו-NEW ===
    ori_options = [o for o in options if o.key.startswith("ORI_")]
    new_options = [o for o in options if not o.key.startswith("ORI_")]

    # === אילוצי ORI ו-NEW לגבי סכום ===
    for o in options:
        if o in ori_options:
            # מסלולי ORI – אם y=1 → x = סכום מקורי קבוע
            FIXED_AMOUNT = float(o.original_principal)
            prob += x[o.key] <= FIXED_AMOUNT * y[o.key]
            prob += x[o.key] >= FIXED_AMOUNT * y[o.key]
        else:
            # מסלולים חדשים – אין מינימום כאן (מטופל לפי סוג בהמשך)
            prob += x[o.key] <= BIG_M * y[o.key]
            prob += x[o.key] <= (con.MAX_SHARE_PER_OPTION * loan_amount) * y[o.key]


    # === מגבלת סוגי ריבית (z_t) רק למסלולים חדשים ===
    types_new = sorted(set(o.rate_type for o in new_options))
    z = {t: pl.LpVariable(f"z_{t}", cat="Binary") for t in types_new}

    for o in new_options:
        prob += y[o.key] <= z[o.rate_type]

    # לפחות 3–4 סוגי ריבית מתוך חדשים בלבד
    if len(types_new) >= 4:
        prob += 3 <= pl.lpSum(z[t] for t in types_new) <= 4
    elif len(types_new) >= 3:
        prob += pl.lpSum(z[t] for t in types_new) >= 3

    # === דרישה אסימטרית למינימום (חלק קריטי) ===

    types_all = sorted(set(o.rate_type for o in options))

    ori_by_type = {t: [o for o in ori_options if o.rate_type == t] for t in types_all}
    new_by_type = {t: [o for o in new_options if o.rate_type == t] for t in types_all}

    for t in types_all:
        ori_exist = len(ori_by_type[t]) > 0

        if not ori_exist:
            # אין מסלול ORI מהסוג הזה → מסלולים חדשים חייבים מינימום
            for o in new_by_type[t]:
                prob += x[o.key] >= con.MIN_ACTIVE_SHARE * loan_amount * y[o.key]
        else:
            # יש ORI מהסוג הזה → NEW מותר להיות בכל גודל, גם קטן מ-MIN_ACTIVE_SHARE
            pass


    # === מגבלת פריים ===
    prob += pl.lpSum(x[o.key] for o in options if o.rate_type == "פריים") >= 0.10 * loan_amount
    #prob += pl.lpSum(x[o.key] for o in options if o.rate_type == "פריים") <= 0.40 * loan_amount

    # === פונקציית מטרה ===
    prob += pl.lpSum(eff_cost[k] * x[k] for k in x)

    # === סכום כספי כולל ===
    prob += pl.lpSum(x[k] for k in x) == loan_amount

    # === חשיפות ===
    prob += pl.lpSum(x[o.key] for o in options if _is_fixed(o.rate_type)) >= min_fixed_share * loan_amount
    prob += pl.lpSum(x[o.key] for o in options if _is_variable(o.rate_type)) <= max_variable_share * loan_amount
    prob += pl.lpSum(x[o.key] for o in options if _is_fixed(o.rate_type)) <= 0.5 * loan_amount
    prob += pl.lpSum(x[o.key] for o in options if _is_cpi(o.rate_type)) <= 0.5 * loan_amount
    prob += pl.lpSum(x[o.key] for o in options if o.rate_type == "פריים") <= 0.4 * loan_amount

    # === מגבלת מספר מסלולים (רק חדשים) ===
    if len(new_options) >= 4:
        prob += 3 <= pl.lpSum(y[o.key] for o in new_options) <= 4
    elif len(new_options) >= 3:
        prob += pl.lpSum(y[o.key] for o in new_options) >= 3

    # === החזר חודשי ראשון ===
    pmt1_vals = {o.key: float(o.pmt1_per_unit) for o in options}
    prob += pl.lpSum(pmt1_vals[o.key] * (x[o.key] / loan_amount) for o in options) <= max_monthly_payment

    # === החזר מקסימלי לאורך חיי ההלוואה ===
    max_pmt_all = {o.key: float(o.max_payment_in_all_mortage) for o in options}
    prob += pl.lpSum(max_pmt_all[o.key] * (x[o.key] / loan_amount) for o in options) <= 1.18 * max_monthly_payment

    # === משתני אחוזים שלמים ===
    p = {
        o.key: pl.LpVariable(f"p_{o.key}", lowBound=0, upBound=100, cat="Integer")
        for o in options
    }

    for o in options:
        prob += x[o.key] == (p[o.key] / 100) * loan_amount

    prob += pl.lpSum(p[o.key] for o in options) == 100

    #import pdb;pdb.set_trace()
    
    # === DEBUG LOGGING SETUP ===
    debug_payload = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "loan_amount": loan_amount,
            "sensitivity": sensitivity,
            "prepay": prepay_window_key,
            "objective": objective_mode,
            "alpha": alpha,
            "max_pmt": max_monthly_payment
        },
        "candidates": [],
        "result": "Pending"
    }
    
    # Pre-calculate candidate debug info
    cand_map = {}
    for o in options:
        # Re-calc discount for logging
        d_val = 0.0
        if _is_fixed(o.rate_type):
            d_val = con.SENSITIVITY_DISCOUNT.get(sensitivity, 0)/100.0 + \
                    con.PREPAY_DISCOUNT.get(prepay_window_key, {}).get("קבועה", 0)/100.0
        else:
             sub_dict = con.PREPAY_DISCOUNT.get(prepay_window_key, {}).get("משתנה", 0)
             if isinstance(sub_dict, dict):
                 d_val = sub_dict.get(o.freq_rate_change, 0)/100.0
             else:
                 d_val = float(sub_dict)/100.0

        c_info = {
            "key": o.key,
            "type": o.rate_type,
            "months": o.months,
            "rate": o.annual_rate,
            "cost": o.cost_per_unit,
            "pmt1": o.pmt1_per_unit,
            "base_norm": base_norm.get(o.key, 0),
            "pmt1_norm": pmt1_norm.get(o.key, 0),
            "discount": d_val,
            "eff_cost": eff_cost.get(o.key, 0),
            "selected": False,
            "amount": 0.0
        }
        debug_payload["candidates"].append(c_info)
        cand_map[o.key] = c_info

    # === פתרון ===
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=con.TIME_LIMIT, gapRel=0.01)
    #solver = pl.PULP_CBC_CMD(msg=False, timeLimit=10, gapRel=0.01)
    prob.solve(solver)
    
    status_str = pl.LpStatus[prob.status]
    debug_payload["result"] = status_str

    if status_str != "Optimal":
        return None, debug_payload

    alloc = {k: float(x[k].value() or 0.0) for k in x}
    alloc_n = {k: v for k, v in alloc.items() if v > 0}
    
    # Update debug logs with selection
    for k, v in alloc_n.items():
        if k in cand_map:
            cand_map[k]["selected"] = True
            cand_map[k]["amount"] = v
            
    debug_payload["allocation"] = alloc_n
    
    return alloc_n, debug_payload

# ---------- פונקציית על – אין תלות ב-tracks ----------
def optimize_mortgage(
    calculate_schedule: Callable,
    summarize_schedule: Callable,
    *,
    loan_amount: float,
    ltv_input: str, 
    monthly_income_net: float,
    sensitivity: str = "בינוני",
    prepay_window_key: str = "לא",
    durations_months: Optional[List[int]] = None,
    bank_of_israel_rate: float = con.bank_of_israel_rate,
    prime_margin: float = con.prime_margin,
    objective_mode: str = "balanced",
    alpha: float = 0.5,
    max_monthly_payment: float = 6000,
    routes_data_ori = None,
    current_prime_rate_for_comparison: float = 4.0 # ברירת מחדל אם לא סופק
) -> Tuple[Optional[List[Dict]], Optional[Dict], Optional[str]]:
    
    ok, errs, derived = check_constraints(
        loan_amount=loan_amount,
        ltv_input=ltv_input,
        monthly_income_net=monthly_income_net,
        sensitivity=sensitivity,
        max_monthly_payment = max_monthly_payment,
    )
    if not ok:
        return None, None, " | ".join(errs)
    
    loan_pct_bucket = ltv_input # כבר כתוית ("35%"/"50%"...)
    
    rate_provider = _make_anchor_rate_provider(bank_of_israel_rate, prime_margin, loan_pct_bucket)
    # מועמדים – תמיד מהקטלוג הפנימי בלבד
    options = generate_candidate_mixtures(
        calculate_schedule,
        summarize_schedule,
        durations_months,
        rate_provider,
        ref_principal=float(loan_amount),
        loan_pct_bucket=loan_pct_bucket,
        routes_data_ori = routes_data_ori
    )
    if not options:
        return None, None, "לא נמצאו אופציות חוקיות לחישוב (בדוק מקור ריביות/עוגנים)."
    # פתרון
    alloc = None
    debug_info = None
    alloc, debug_info = _solve_lp(
        options, loan_amount,
        derived["min_fixed_share"], derived["max_variable_share"],
        prepay_window_key,
        objective_mode=objective_mode, alpha=alpha,
        sensitivity=str(sensitivity),max_monthly_payment=max_monthly_payment
    )
    print(alloc)

    if alloc is None:
        return None, {"debug_data": debug_info} if debug_info else None, "לא ניתן להרכיב תמהיל תחת המגבלות."
    # הרכבת תוצאה
    opt_by_key = {o.key: o for o in options}
    selected: List[Dict] = []
    final_allocations = [] # List to store rounded allocations
    total_interest = 0.0
    total_indexation = 0.0
    pmt1_total = 0.0
    for key, amount in alloc.items():
        o = opt_by_key[key]
        
        # With p as Integer, amount should be integer (or very close float)
        # amount = (p/100) * loan
        
        amount_val = round(float(amount))
        if amount_val > 0:
             final_allocations.append({"key": key, "amount": amount_val, "opt": o})

    # 2. Fix sum discrepancy (just in case of tiny floating point drifts)
    current_sum = sum(item["amount"] for item in final_allocations)
    diff = int(loan_amount) - current_sum
    
    if diff != 0 and final_allocations:
        # Add difference to the track with the largest amount
        final_allocations.sort(key=lambda x: x["amount"], reverse=True)
        final_allocations[0]["amount"] += diff

    for item in final_allocations:
        o = item["opt"]
        amount = item["amount"]

        kwargs = dict(
            principal=float(amount),
            months=int(o.months),
            annual_rate=float(o.annual_rate),
            schedule_type=o.schedule_type,
            rate_type=o.rate_type,
            freq_rate_change=int(o.freq_rate_change),
        )

        # --- Global Manual Interest Increase Check ---
        # If set in config, boost the rate multiplicatively
        if getattr(con, 'manual_interest_increase', 0.0) > 0:
            boost_factor = 1 + (con.manual_interest_increase / 100.0)
            kwargs['annual_rate'] = kwargs['annual_rate'] * boost_factor

        if o.rate_type == "פריים":
            if "ORI" in o.key:
                prime_offset =  o.annual_rate - (con.prime_margin + con.bank_of_israel_rate)
            else: 
                prime_offset = con.LOAN_ADJ_RULES["Prime"][loan_pct_bucket] + con.SPREADS['Prime']
            kwargs["prime_margin"] = con.prime_margin   
            kwargs["prime_offset"] = prime_offset

        sch = calculate_schedule(**kwargs)
        pr, it, idx = summarize_schedule(sch)
        total_interest += float(it)
        total_indexation += float(idx)
        pmt1_total += float(sch[0][2]) if sch else 0.0
        selected.append({
            "principal": float(amount),
            "months": int(o.months),
            "rate": kwargs['annual_rate'], # Use the (potentially boosted) rate used for calculation
            "rate_type": o.rate_type,
            "freq": int(o.freq_rate_change),
            "schedule_t": o.schedule_type,
            "schedule": sch,
        })
    total_payment = loan_amount + total_interest + total_indexation
    totals = {
        "total_payment": float(total_payment),
        "total_cost": float(total_payment),
        "total_interest": float(total_interest),
        "total_indexation": float(total_indexation),
        "pmt1": float(pmt1_total),
        "debug_data": debug_info
    }

    return selected, totals, None

# ---------- עזר לגרף ----------
def _aggregate_monthly_payment(schedules: List[List[List[float]]]) -> Tuple[List[int], List[float]]:
    max_m = 0
    for sch in schedules:
        if sch: max_m = max(max_m, int(sch[-1][0]))
    months = list(range(1, max_m + 1))
    totals = [0.0] * len(months)
    for sch in schedules:
        for row in sch:
            m = int(row[0]); pay = float(row[2])
            totals[m - 1] += pay
    return months, totals


def run_optimize(loan_amount, first_payment, max_months,capital_allocation,routes_data):
    return optimize_mortgage(
        calculate_schedule,
        summarize_schedule,
        loan_amount=float(loan_amount),
        ltv_input=capital_allocation,
        monthly_income_net=float(first_payment * 3.5),
        sensitivity="בינוני",
        prepay_window_key="לא",
        durations_months=list(range(12, max_months, 1)),
        bank_of_israel_rate=con.bank_of_israel_rate,
        prime_margin=con.prime_margin,
        objective_mode="balanced",
        alpha=0.5,
        max_monthly_payment=first_payment,
        routes_data_ori = routes_data,
    )




def create_4_candidate_mortages(raw_PDF_tables,capital_allocation):
    #import pdb;pdb.set_trace()
    #print('raw_PDF_tables:',raw_PDF_tables)
    mortgage_route = create_total_months(create_interest_type(raw_PDF_tables))
    routes_data, update_data, non_indx_data, optimal_data = [], [], [], []
    loan_amount, first_payment, max_months = 0, 0, 0
    clac_rate_int = InterestRateCalculator()
    # ------------------
    for i, rec in mortgage_route.items():
        
        #print(rec)
        loan_amount += rec['Payoff_Amount']# was Payoff_remind
        max_months = max(max_months, rec['total_months_Remained'])
        ### current
        
        if 1:
            months_to_next_reset = rec.get('months_to_next_reset', 0) or 0
            rate_type = rec.get('current_interest_type')
            #import pdb;pdb.set_trace()
            #months_to_next_reset = 0
            sch = calculate_schedule(
                rec['Payoff_Amount'],# was Payoff_remind
                rec['total_months_Remained'],
                rec['Nominal_Interest_Rate_Date_of_Letter'], 
                rec['Loan_Repayment_Method'], 
                rec['current_interest_type'],
                rec['Interest_Rate_Change_Frequency_in_Months'],
                prime_margin=con.prime_margin, 
                prime_offset=rec['Nominal_Interest_Rate_Date_of_Letter'] - (con.bank_of_israel_rate + con.prime_margin) if rate_type == 'פריים' else 0,
                months_to_next_reset=months_to_next_reset,
            )
            print('current current current')
            prime_offset_temp=rec['Nominal_Interest_Rate_Date_of_Letter'] - (con.bank_of_israel_rate + con.prime_margin) if rate_type == 'פריים' else 0
            print(rec['Payoff_Amount'], rec['total_months_Remained'],rec['Nominal_Interest_Rate_Date_of_Letter'], rec['Loan_Repayment_Method'], rec['current_interest_type'],rec['Interest_Rate_Change_Frequency_in_Months'],prime_offset_temp,months_to_next_reset)
            # סיכום עבור כל מסלול
            P,I,K = summarize_schedule(sch)
            routes_data.append([{
                "מסלול": rate_type,
                "סכום": rec['Payoff_Amount'],# was Payoff_remind
                "תקופה (חודשים)": rec['total_months_Remained'],
                "ריבית": rec['Nominal_Interest_Rate_Date_of_Letter'],
                "החזר ראשון": round(sch[0][2],0),
                "תדירות שינוי": rec['Interest_Rate_Change_Frequency_in_Months'],
                "סה״כ ריבית והצמדה": I+K,
                'סך כל התשלומים': P+I+K,
                'החזר בשיא': max (ii[2] for ii in sch),
                'סוג מסלול': rec['Loan_Repayment_Method'],
                'internal_rate_of_return': rec['internal_rate_of_return']

                },
                sch])
                
            
            first_payment += sch[0][2]
        
        ### update mortage ###
        if 1:
            #import pdb;pdb.set_trace()
            months_to_next_reset = rec.get('months_to_next_reset', 0) or 0
            rate_type = rec.get('current_interest_type')
            
            # ------------------------------------------------------------------
            # Smart Update Logic:
            # אנחנו משווים בין:
            # 1. הריבית הנוכחית שלך (current_rate)
            # 2. ריבית השוק הנוכחית (market_rate_derived)
            # ולוקחים את המינימום מבניהם לתרחיש "משכנתא מעודכנת".
            
            # שלב א: חישוב ריבית השוק (market_rate_derived)
            market_adj_rate = clac_rate_int.get_adjusted_rate(
                con.ANCHOR_TRACK_MAP[rate_type],
                capital_allocation,
                rec['Interest_Rate_Change_Frequency_in_Months'],
                rec['total_months_Remained'], 
            )
            
            if rate_type == 'פריים':
                # עבור פריים: ריבית שוק = פריים בנק ישראל + מרווח רגולטורי/שוק
                market_base_prime = clac_rate_int.prime_rate
                market_prime_spread = clac_rate_int.Rules_loan_percentage["Prime"][capital_allocation] + clac_rate_int.Spread['Prime']
                market_rate_derived = market_base_prime + market_prime_spread
            else:
                # עבור שאר המסלולים: מה שמגיע מהמחשבון
                market_rate_derived = market_adj_rate

            # שלב ב: שליפת הריבית הנוכחית של המשתמש
            user_current_rate = rec['Nominal_Interest_Rate_Date_of_Letter']

            # שלב ג: בחירת הריבית הטובה מבין השתיים
            if user_current_rate < market_rate_derived:
                # הריבית המקורית טובה יותר -> נשארים איתה
                final_update_rate = user_current_rate
                is_user_rate_kept = True
            else:
                # ריבית השוק טובה יותר (או שווה) -> מעדכנים לשוק
                final_update_rate = market_rate_derived
                is_user_rate_kept = False

            # שלב ד: חישוב Prime Offset (רלוונטי רק לפריים)
            if rate_type == 'פריים':
                if is_user_rate_kept:
                    # שומרים על המרווח המקורי
                    # חישוב: ריבית משתמש פחות פריים בסיס של היום (הנחה: הפריים המצוטט הוא נכון להיום)
                    # הערה: אם Nominal_Interest_Rate_Date_of_Letter היא היסטורית, זה עלול להיות לא מדויק,
                    # אבל זו האינדיקציה הכי טובה ל"כמה המשתמש משלם היום".
                    # נניח ש-Nominal... היא הריבית האבסולוטית העדכנית.
                    current_spread = user_current_rate - (con.bank_of_israel_rate + con.prime_margin)
                    prime_offset = current_spread
                else:
                    # לוקחים מרווח שוק
                    prime_offset = market_prime_spread
            else:
                prime_offset = 0 # לא בשימוש במסלולים אחרים
            
            # עדכון המשתנה update_rate לשימוש בהמשך (אם יש תלויות אחרות בקוד)
            update_rate = final_update_rate
                
            sch = calculate_schedule(
                rec['Payoff_Amount'],# was Payoff_remind 
                rec['total_months_Remained'],
                update_rate,
                rec['Loan_Repayment_Method'], 
                rec['current_interest_type'],
                rec['Interest_Rate_Change_Frequency_in_Months'],
                prime_margin= con.prime_margin, #1.5, 
                prime_offset=prime_offset,
                months_to_next_reset=months_to_next_reset
            )
            # סיכום עבור כל מסלול
            P,I,K = summarize_schedule(sch)
            update_data.append([{
                "מסלול": rate_type,
                "סכום": rec['Payoff_Amount'],# was Payoff_remind
                "תקופה (חודשים)": rec['total_months_Remained'],
                "ריבית": update_rate,
                "החזר ראשון": round(sch[0][2],0),
                "תדירות שינוי": rec['Interest_Rate_Change_Frequency_in_Months'],
                "סה״כ ריבית והצמדה": I+K,
                'סך כל התשלומים': P+I+K,
                'החזר בשיא': max (ii[2] for ii in sch),
                'סוג מסלול': rec['Loan_Repayment_Method']},
                sch])
            
        ### non-indx data ###
        if 1:
            rate_type = rec.get('current_interest_type')
            months_to_next_reset = rec.get('months_to_next_reset', 0) or 0
            ori_rate = rec['Nominal_Interest_Rate_Date_of_Letter']

            if rate_type not in ['מצ','קצ']:
                sch = calculate_schedule(
                rec['Payoff_Amount'],# was Payoff_remind
                rec['total_months_Remained'],
                rec['Nominal_Interest_Rate_Date_of_Letter'], 
                rec['Loan_Repayment_Method'], 
                rec['current_interest_type'],
                rec['Interest_Rate_Change_Frequency_in_Months'],
                prime_margin=con.prime_margin, 
                prime_offset=rec['Nominal_Interest_Rate_Date_of_Letter'] - (con.prime_margin + con.bank_of_israel_rate),
                months_to_next_reset=months_to_next_reset,
                )
                final_rate_type = rate_type
                final_rate = ori_rate
            else:
            
                if rate_type == 'מצ':
                    rate_type_update = "מלצ"
                    
                elif rate_type == 'קצ':
                    rate_type_update = "קלצ"
                
                update_rate = clac_rate_int.get_adjusted_rate(
                        con.ANCHOR_TRACK_MAP[rate_type_update],
                        capital_allocation,
                        rec['Interest_Rate_Change_Frequency_in_Months'],
                        rec['total_months_Remained'], 
                    )
                print('^^^^^^^^^^^^^^^^')
                print(f'rate_type:{rate_type},ori_rate:{ori_rate}')
                print(f'rate_type:{rate_type_update},update_rate:{update_rate}')
                #if update_rate<ori_rate :
                final_rate = update_rate
                final_rate_type = rate_type_update
                
                #else:
                #    final_rate = ori_rate
                #    final_rate_type = rate_type
                       
                sch = calculate_schedule(
                    rec['Payoff_Amount'],# was Payoff_remind
                    rec['total_months_Remained'],
                    final_rate, 
                    rec['Loan_Repayment_Method'], 
                    final_rate_type,
                    rec['Interest_Rate_Change_Frequency_in_Months'],
                    prime_margin=con.prime_margin, #1.5, 
                    prime_offset=rec.get('prime_offset', 0) if final_rate_type == 'פריים' else 0,
                    months_to_next_reset=months_to_next_reset
                )
            # סיכום עבור כל מסלול
            P,I,K = summarize_schedule(sch)
            non_indx_data.append([
                {"מסלול": final_rate_type,
                "סכום": rec['Payoff_Amount'],# was Payoff_remind
                "תקופה (חודשים)": rec['total_months_Remained'],
                "ריבית": final_rate,
                "החזר ראשון": round(sch[0][2],0),
                "תדירות שינוי": rec['Interest_Rate_Change_Frequency_in_Months'],
                "סה״כ ריבית והצמדה": I+K,
                'סך כל התשלומים': P+I+K,
                'החזר בשיא': max (ii[2] for ii in sch),
                'סוג מסלול': rec['Loan_Repayment_Method']},
                sch])
        
    # --- אופטימיזציה ---
    sol_tracks, totals, err = run_optimize(loan_amount, first_payment, max_months,capital_allocation,routes_data)
    
    for track in sol_tracks:
        principal = track['principal']
        months = track['months']
        rate = track['rate']
        rate_type = track['rate_type']
        freq = track['freq']
        sch = track['schedule']
        P,I,K = summarize_schedule(sch)

        optimal_data.append([
            {"מסלול": rate_type,
            "סכום": principal,
            "תקופה (חודשים)": months,
            "ריבית": rate,
            "החזר ראשון": round(sch[0][2],0),
            "תדירות שינוי": freq,
            "סה״כ ריבית והצמדה": I+K,
            'סך כל התשלומים': P+I+K,
            'החזר בשיא': max (ii[2] for ii in sch)},
            sch])
    
    # ------------------
    return (routes_data, update_data, non_indx_data, optimal_data)

def find_best_mortage(raw_PDF_tables,capital_allocation):
    ori_routes_mortage, update_mortage, non_indx_mortage, optimal_mortage = create_4_candidate_mortages(raw_PDF_tables,capital_allocation)
    #print(ori_routes_mortage)
    #import pdb ; pdb.set_trace()
    total_payments_ori = sum([root['סך כל התשלומים'] for root,ss in ori_routes_mortage])
    first_repaiment_ori = sum([root["החזר ראשון"] for root,ss in ori_routes_mortage])
    max_repaiment_ori = sum([root["החזר בשיא"] for root,ss in ori_routes_mortage])
    
    # 1. סינון ראשוני של כל המועמדים לפי תנאי הסף (החזר חודשי וחיסכון מינימלי)
    internal_candidates = []
    optimal_candidate = None

    for mortage_candidate, name in [(update_mortage, 'update_mortage'), 
                                     (non_indx_mortage, 'non_indx_mortage'), 
                                     (optimal_mortage, 'optimal_mortage')]:
        
        total_payments = sum([root['סך כל התשלומים'] for root, ss in mortage_candidate])
        first_repaiment = sum([root["החזר ראשון"] for root, ss in mortage_candidate])
        max_repaiment = sum([root["החזר בשיא"] for root, ss in mortage_candidate])
        
        # בדיקת תנאי סף בסיסיים (החזר חודשי וחיסכון מינימלי מול המקור)
        if first_repaiment - first_repaiment_ori > con.max_diff_of_first_payment:
            continue
        if max_repaiment - max_repaiment_ori > con.max_diff_of_max_payment:
            continue
        if total_payments + con.no_savings > total_payments_ori:
            continue
            
        candidate_data = [mortage_candidate, total_payments, first_repaiment, max_repaiment, name]
        
        if name == 'optimal_mortage':
            optimal_candidate = candidate_data
        else:
            internal_candidates.append(candidate_data)

    # 2. בחירת התמהיל המנצח לפי העדפה פנימית
    best_candidate = None
    
    # מציאת התמהיל הפנימי הכי זול (אם יש כזה)
    best_internal = min(internal_candidates, key=lambda x: x[1]) if internal_candidates else None

    if optimal_candidate:
        if best_internal:
            # השוואה יחסית: האם האופטימלי חוסך 40,000 ש"ח יותר מהפנימי הכי טוב?
            # (זכור: total_payments נמוך יותר זה טוב יותר)
            if best_internal[1] - optimal_candidate[1] >= 40000:
                best_candidate = optimal_candidate
            else:
                best_candidate = best_internal
        else:
            # אין תמהיל פנימי שעבר את התנאים, בודקים אם האופטימלי חוסך מספיק מול המקור
            
            best_candidate = optimal_candidate
    else:
        # אין תמהיל אופטימלי רלוונטי, בוחרים את הפנימי הכי טוב
        best_candidate = best_internal

    
    return best_candidate



def weighted_avg_rate_fun(group):
    if not group:
        return None
    total_amount = sum(x[0] for x in group)
    weighted_sum = sum(x[0] * x[1] for x in group)
    return total_amount,weighted_sum / total_amount if total_amount > 0 else None

def compute_summary(routes):
    total_amount = 0
    all_refunds=0
    weighted_sum = 0
    max_term = 0
    Fixed = []
    Variable = []
    Prime = []
    first_peyment = 0
    for route, sch in routes:
        amount = route["סכום"]
        rate = route["ריבית"]
        term = route["תקופה (חודשים)"]
        rate_type = route["מסלול"]
        print(rate_type)
        all_refunds+=route['סך כל התשלומים']
        first_peyment+= route['החזר ראשון']

        if rate_type == 'מצ' or rate_type == 'מלצ':
            Variable.append([amount,rate])
        elif rate_type == 'קצ' or rate_type == 'קלצ':
            Fixed.append([amount,rate])
        else:
            Prime.append([amount,rate])
        
        if amount is not None and rate is not None:
            total_amount += amount
            weighted_sum += amount * rate
        
        if term is not None and term > max_term:
            max_term = term
    #import pdb ; pdb.set_trace()
    weighted_avg_rate = weighted_sum / total_amount if total_amount > 0 else None
    Refund_amount_per_shekel = all_refunds/total_amount
    
    
    data = {'weighted_avg_rate':weighted_avg_rate,
             'max_term':max_term,
             'Refund_amount_per_shekel':Refund_amount_per_shekel,
             'prime_summery':weighted_avg_rate_fun(Prime),
             'Fixed_summery':weighted_avg_rate_fun(Fixed),
             'Variable_summery':weighted_avg_rate_fun(Variable),
             'first_peyment':first_peyment,

             }
    
    return data

def safe_name(name: str) -> str:
        """ניקוי שם קובץ מתווים בעייתיים"""
        return re.sub(r"[^A-Za-z0-9_.\-א-ת]", "_", name)

def _schedule_arrays(sch: List[List[float]]):
    xs = [row[0] for row in sch]
    opening = [row[1] for row in sch]
    payment = [row[2] for row in sch]
    interest = [row[3] for row in sch]
    principal_base = [row[5] for row in sch]
    indexation = [row[6] for row in sch]
    closing = [row[7] for row in sch]
    used_rate = [row[8] for row in sch]
    return xs, opening, payment, interest, principal_base, indexation, closing, used_rate

def _pad(seq: List[float], L: int, pad_with_zero: bool = True) -> List[float]:
    if len(seq) >= L:
        return seq[:L]
    pad_val = 0.0 if pad_with_zero else (seq[-1] if seq else 0.0)
    return seq + [pad_val] * (L - len(seq))


# --- מיפוי טיפוס מסלול (קוד בנק) -> טיפוס ריבית טקסטואלי ישן שלך ---
def _map_interest_type_from_loan_type(loan_type: dict) -> str: 
    """
    קלט: רשומת track אחת מתוך data["tracks"].
    פלט: מחרוזת Hebrew כמו 'פריים', 'העובק', 'הנתשמ' וכו' לשדה Interest_Type.
    
    חשוב: כאן צריך להשלים את המיפוי לפי הגדרת הבנק שלך.
    כרגע זה רק שלד.
    """
    # Indexation_Basis: 3,5,9
    if loan_type == 1:
        return "פריים"
    elif loan_type == 2:
        return "קלצ"       
    elif loan_type == 3: 
        return "קצ"       
    elif loan_type == 4: 
        return "מלצ"       
    elif loan_type == 5:
        return "מצ" 
    elif loan_type == 6:
        return "מטח דולר"      
    elif loan_type == 7: 
        return "מטח יורו"     
    elif loan_type == 8: 
        return "מקמ"      
    elif loan_type == 9:
        return "זכאות"      
    elif loan_type == 11:
        return "מענק"



    else:
        print(f'loan_type:{loan_type}')
        return "Unknown"


def _map_indexation_basis(track: dict) -> bool:
    """Indexation_Basis: True אם יש הצמדה, אחרת False."""
    infl = track.get("loan_value_inflation", 0) or 0.0
    return bool(infl and abs(float(infl)) > 0)


def _map_loan_repayment_method(board: int) -> str: 
    """
    Loan_Repayment_Method: אצלך היה 'רציפש' (שפיצר).
    בקובץ JSON יש loan_board (1,2,...).
    כאן צריך להשלים לפי הפרשנות שלך לטבלת הקידוד של הבנק.
    כרגע הנחה: 1 = שפיצר.
    """
     
    if board == 1:
        return "רציפש"   # שפיצר
    

    elif board == 2: return "קרן שווה"
    elif board == 3: return "בלון מלא"
    elif board == 4: return "בלון חלקי"
    elif board == 5: return "כפי יכולתך"
    elif board == 8: return "משכנתה בטוחה"
    


    return "רציפש"


def _parse_date_or_nat(s: str):
    if not s:
        return None
    return pd.to_datetime(s, dayfirst=True, errors="coerce")





def convert_api_json_to_loan_tracks(api_json: dict) -> dict:
    """
    ממפה את מבנה ה־JSON החדש (עם data/tracks) לפורמט loan_tracks
    שבו השתמשת עד עכשיו.
    """
    data = api_json.get("data", {})
    date_balance_str = data.get("date_balance")
    latter_date = _parse_date_or_nat(date_balance_str)

    loan_tracks: dict[int, dict] = {}

    for track in data.get("tracks", []):
        loan_tracks[track['internal_id']] = {}

        loan_tracks[track['internal_id']]['Loan_number'] = track['track_bank_number']
        loan_tracks[track['internal_id']]['Loan_Purpose'] = track['loan_purpose']
        loan_tracks[track['internal_id']]['Original_Loan_Amount'] = float(track['original_loan_value'])
        loan_tracks[track['internal_id']]['Expected_Last_Payment_Date'] = _parse_date_or_nat(track['date_end'])
        loan_tracks[track['internal_id']]['Loan_Type'] = 'קנב' # need to change - Rachel2Blame
        loan_tracks[track['internal_id']]['Loan_Repayment_Method'] = _map_loan_repayment_method(int(track['loan_board']))
        loan_tracks[track['internal_id']]['Interest_Type'] = _map_interest_type_from_loan_type(int(track['loan_type']))
        try:
            loan_tracks[track['internal_id']]['Interest_Rate_Change_Frequency_in_Months'] = int(track['date_change_frequency'])
        except:
            loan_tracks[track['internal_id']]['Interest_Rate_Change_Frequency_in_Months'] = None
        if loan_tracks[track['internal_id']]['Interest_Type'] == "פריים":
            loan_tracks[track['internal_id']]['Interest_Rate_Change_Frequency_in_Months'] = 1

        loan_tracks[track['internal_id']]['latter_date_object'] = latter_date
        loan_tracks[track['internal_id']]['Nominal_Interest_Rate_Date_of_Letter'] = float(track['loan_interest'])
        loan_tracks[track['internal_id']]['Next_Interest_Rate_Change_Date'] = _parse_date_or_nat(track['date_change_interest'])
        if int(track['loan_type'])  in [3,5,9]:
            loan_tracks[track['internal_id']]['Indexation_Basis'] = True
        else:
            loan_tracks[track['internal_id']]['Indexation_Basis'] = False
        loan_tracks[track['internal_id']]['principal_remind'] = float(track['loan_value'])
        loan_tracks[track['internal_id']]['Principal_Indexation'] = float(track['loan_value_inflation'])
        loan_tracks[track['internal_id']]['Prepayment_Fee'] = float(track['fee_differences'])
        
        loan_tracks[track['internal_id']]['Payoff_remind'] = float(track['loan_value'])+float(track['loan_value_inflation'])+float(track['fee_differences'])
        
        loan_tracks[track['internal_id']]['Payoff_Amount'] = float(track['loan_value_interest'])+loan_tracks[track['internal_id']]['Payoff_remind']

        loan_tracks[track['internal_id']]['First_Payment_Date'] = _parse_date_or_nat(track['date_end']) - pd.DateOffset(months=int(track['loan_years']))
        loan_tracks[track['internal_id']]['internal_rate_of_return'] = float(track['internal_rate_of_return'])# internal_rate_of_return
    #print('@@@@@@@@@@@')
    #print(loan_tracks)
    #print('@@@@@@@@@@@')
    return loan_tracks

def convert_api_json_to_first_loan_tracks_old(api_json: dict) -> dict:

    data = api_json.get("data", {})
    first_loan_tracks: dict[int, dict] = {}
    approval_date = data['approval_date']
    purpose = None
    if 'purpose' in data: 
        purpose=data['purpose']

    i=0
    for track in data.get("tracks", []):
        i+=1
        first_loan_tracks[track[i]]={}
        temp = first_loan_tracks[track[i]]
        temp['Interest_Type'] = _map_interest_type_from_loan_type(int(track['loan_type']))
        temp['Loan_Repayment_Method'] = _map_loan_repayment_method(int(track['loan_board']))
        temp['loan_months'] = int(track['loan_years'])
        temp ['principal_remind'] = float(track['loan_value'])
        try:
            temp['Interest_Rate_Change_Frequency_in_Months'] = int(track['date_change_frequency'])
        except:
            temp['Interest_Rate_Change_Frequency_in_Months'] = None
        if temp['Interest_Type'] == "פריים":
            temp['Interest_Rate_Change_Frequency_in_Months'] = 1
        temp['Nominal_Interest_Rate_Date_of_Letter'] = float(track['loan_interest'])
        
        if 'track_purpose' in track:
            temp['propose']= track['track_purpose']
        elif not purpose:
            temp['propose']= purpose
        

    return first_loan_tracks

def convert_api_json_to_first_loan_tracks(api_json: dict) -> dict:
    
    data = api_json.get("data", {})
    first_loan_tracks=[]
    approval_date = data['approval_date']
    purpose = None
    if 'purpose' in data: 
        purpose=data['purpose']

    #i=0
    for track in data.get("tracks", []):
        #i+=1
        #first_loan_tracks[i]={}
        temp = {}
        temp['rate_type'] = _map_interest_type_from_loan_type(int(track['loan_type']))
        temp['schedule_t'] = _map_loan_repayment_method(int(track['loan_board']))
        temp['months'] = int(track['loan_years'])
        temp ['principal'] = float(track['loan_value'])
        try:
            temp['freq'] = int(track['date_change_frequency'])
        except:
            temp['freq'] = None
        if temp['rate_type'] == "פריים":
            temp['freq'] = 1
        temp['rate'] = float(track['loan_interest'])
        
        if 'track_purpose' in track:
            temp['propose']= track['track_purpose']
        elif not purpose:
            temp['propose']= purpose
        first_loan_tracks.append(temp)

    return first_loan_tracks