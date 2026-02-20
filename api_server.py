from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
import json
import importlib
import os
import pprint
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from functions import (
    calculate_schedule, summarize_schedule, optimize_mortgage, 
    create_4_candidate_mortages, convert_api_json_to_loan_tracks,
    convert_api_json_to_first_loan_tracks, InterestRateCalculator,
    _schedule_arrays,_aggregate_monthly_payment,aggregate_yearly,
    find_best_mortage
)
import config as con


app = FastAPI(title="Mortgage API Server")

CONFIG_SYNC_KEY_ENV = "CALC_API_SYNC_KEY"


class ConfigSyncRequest(BaseModel):
    updates: Dict[str, Any]


def _apply_config_updates(config_path: Path, updates: Dict[str, Any]) -> None:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config file: {e}")

    for key, value in updates.items():
        escaped_key = re.escape(str(key))
        if isinstance(value, (dict, list)):
            formatted_value = pprint.pformat(value, width=100, sort_dicts=False)
            prefix_pattern = rf'(^|\n){escaped_key}\s*(?::[^=]+)?=\s*'
            p_match = re.search(prefix_pattern, content)
            if p_match:
                start_idx = p_match.end()
                while start_idx < len(content) and content[start_idx].isspace():
                    start_idx += 1
                if start_idx >= len(content):
                    continue
                open_char = content[start_idx]
                close_char = '}' if open_char == '{' else ']'
                if open_char not in "{[":
                    continue

                cnt = 0
                end_idx = start_idx
                in_struct = False
                for i in range(start_idx, len(content)):
                    ch = content[i]
                    if ch == open_char:
                        cnt += 1
                        in_struct = True
                    elif ch == close_char:
                        cnt -= 1

                    if in_struct and cnt == 0:
                        end_idx = i + 1
                        break

                if end_idx > start_idx:
                    content = content[:start_idx] + formatted_value + content[end_idx:]
        else:
            if isinstance(value, str):
                replacement = f'{key} = "{value}"'
                pattern = rf'{escaped_key}\s*=\s*[\'"][^\'"]*[\'"]'
            else:
                replacement = f'{key} = {value}'
                pattern = rf'{escaped_key}\s*=\s*[-+]?\d*\.?\d+'

            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config file: {e}")


def _validate_sync_key(provided_key: Optional[str]) -> None:
    expected_key = os.getenv(CONFIG_SYNC_KEY_ENV, "").strip()
    if not expected_key:
        raise HTTPException(status_code=503, detail="Calculator sync key is not configured")
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid calculator sync key")



class MortgageEngine:
    def __init__(self):
        self.calculator = InterestRateCalculator()

    def refresh_runtime_config(self) -> None:
        # Ensure each request uses the latest values from config.py.
        importlib.reload(con)
        self.calculator = InterestRateCalculator(
            bank_of_israel_rate=float(con.bank_of_israel_rate),
            prime_margin=float(con.prime_margin),
        )
    
    def calculate_ltv_details(self, property_value: float, loan_amount: float) -> dict:
        """
        מחשבת יחס מימון ומדרגת הקצאת הון לפי רגולציית בנק ישראל.
        """
        if loan_amount > 5000000:
            return {"ltv_ratio": 100, "allocation": "100%"}
        if property_value <= 0:
            return {"ltv_ratio": 0, "allocation": "75%"}
        
        ltv_ratio = loan_amount / property_value
        ltv_pct = ltv_ratio * 100
        
        # הגדרת הקצאת הון לפי מדרגות המימון (כפי שמופיע במסמך שלך)
        if ltv_pct <= 45:
            allocation = "35%"
        elif ltv_pct <= 60:
            allocation = "50%"
        elif ltv_pct <= 75:
            allocation = "60%"
        else:
            allocation = "75%"
            
        return {
            "ltv_pct": round(ltv_pct, 2),
            "allocation": allocation
        }
    
    def _calc_max_payment(self, schedules: list) -> float:
        """פונקציית עזר לחישוב שיא ההחזר מכל המסלולים יחד"""
        if not schedules: return 0
        max_m = max(len(s) for s in schedules)
        return max([sum(s[m][2] for s in schedules if m < len(s)) for m in range(max_m)])

    def simulate_manual_tracks(self, tracks_data: List[Dict],property_value) -> Dict:
        """לוגיקה של טאב 1 - סימולטור מלא לכל מסלול ולתמהיל כולו"""
        individual_results = []
        all_schedules = []
        
        loan_amount = sum([i['principal'] for i in tracks_data])
        capital_allocation = self.calculate_ltv_details(property_value,loan_amount)["allocation"] 
        # 1. עיבוד כל מסלול בנפרד
        for i, tr in enumerate(tracks_data):

            if tr['rate'] == 0:
                rate = self.calculator.get_adjusted_rate(
                    con.ANCHOR_TRACK_MAP[tr['rate_type']],
                    capital_allocation,
                    tr['freq'],
                    tr['months'])
            else:
                rate = tr['rate']
                
            # חישוב אופסט לפריים במידה ורלוונטי
            prime_offset = rate - (con.bank_of_israel_rate + con.prime_margin) if tr['rate_type'] == 'פריים' else 0
            
            # יצירת לוח סילוקין
            sch = calculate_schedule(
                tr['principal'], tr['months'], rate, tr['schedule_t'], 
                tr['rate_type'], tr.get('freq'), con.prime_margin, prime_offset
            )
            
            # חילוץ מערכים לגרפים וסיכומים
            xs, opening, payment, interest, principal_base, indexation, closing, used_rate = _schedule_arrays(sch)
            p_sum, i_sum, k_sum = summarize_schedule(sch)
            pr_y, in_y, idx_y = aggregate_yearly(sch) # פונקציה מתוך functions.py

            individual_results.append({
                "id": i + 1,
                "summary": {
                    "principal_paid": p_sum,
                    "interest_paid": i_sum,
                    "indexation_paid": k_sum,
                    "total_repayment": p_sum + i_sum + k_sum,
                    "first_payment": payment[0]
                },
                "graph_data": {
                    "months": xs,
                    "opening_balance": opening,
                    "monthly_payment": payment,
                    "interest_component": interest,
                    "closing_balance": closing,
                    "effective_rate": used_rate,
                    "yearly_distribution": {"principal": pr_y, "interest": in_y, "indexation": idx_y}
                },
                "full_schedule": sch # לטובת הצגת הטבלה
            })
            all_schedules.append(sch)

        # 2. חישוב נתונים מאוחדים (Combined)
        max_l = max(len(s) for s in all_schedules)
        
        # סכימת מערכים עם Padding לאפסים במידה והתקופות שונות
        combined_payment = [sum(s[m][2] for s in all_schedules if m < len(s)) for m in range(max_l)]
        combined_opening = [sum(s[m][1] for s in all_schedules if m < len(s)) for m in range(max_l)]
        combined_interest = [sum(s[m][3] for s in all_schedules if m < len(s)) for m in range(max_l)]
        combined_closing = [sum(s[m][7] for s in all_schedules if m < len(s)) for m in range(max_l)]
        
        # סיכום עלויות כולל
        total_p_sum = sum(r["summary"]["principal_paid"] for r in individual_results)
        total_i_sum = sum(r["summary"]["interest_paid"] for r in individual_results)
        total_k_sum = sum(r["summary"]["indexation_paid"] for r in individual_results)

        return {
            "individual_tracks": individual_results,
            "combined": {
                "summary": {
                    "total_principal": total_p_sum,
                    "total_interest": total_i_sum,
                    "total_indexation": total_k_sum,
                    "total_repayment": total_p_sum + total_i_sum + total_k_sum,
                    "max_payment": max(combined_payment),
                    "first_payment": combined_payment[0]
                },
                "graph_data": {
                    "months": list(range(1, max_l + 1)),
                    "monthly_payment": combined_payment,
                    "opening_balance": combined_opening,
                    "closing_balance": combined_closing,
                    "interest_balance":combined_interest,
                }
            }
        }
    
    def run_optimization(self, params: Dict) -> Dict:
        """לוגיקה של טאב 2 - אופטימיזציה למשכנתא חדשה"""
        
        capital_allocation = self.calculate_ltv_details(params['property_value'],params['loan_amount'])['allocation']
        print(capital_allocation)
        #durations = [y * 12 for y in range(12, int(params['max_scan_years']) + 1)]
        sol, totals, err = optimize_mortgage(
            params['loan_amount'],
            capital_allocation, 
            params['income'],
            con.sensitivity,
            con.prepay_window_key,
            con.durations_months(12*int(params['max_scan_years'])),
            con.bank_of_israel_rate,
            con.prime_margin,
            con.objective_mode,
            con.alpha,
            params['max_pmt']
        )
            
        if err: return {"error": err}
        return {"tracks": sol, "max_payment": self._calc_max_payment([t['schedule'] for t in sol])}

    def analyze_refinance(self, file_content: bytes, property_value: float) -> Dict[str, Any]:
        """
        פונקציה המנתחת את כל תרחישי המחזור ומחזירה מידע מלא להשוואה ופירוט גרפי.
        """
        # 1. עיבוד הקלט וקביעת הקצאת הון (LTV)
        # שימוש בפונקציית העזר הקיימת שלך למיפוי ה-JSON
        
        
        
        
        api_json = json.loads(file_content.decode("utf-8-sig"))
        tracks = convert_api_json_to_loan_tracks(api_json)
        loan_amount = sum([tracks[i]['Payoff_Amount'] for i in tracks.keys()])
        ltv_bucket = self.calculate_ltv_details(property_value,loan_amount)['allocation']
        
        # 2. יצירת 4 תרחישי המחזור (מקור, מעודכנת, לא צמודה, אופטימלית)
        # הפונקציה create_4_candidate_mortages כבר מחשבת לוחות סילוקין (schedules)
        scenarios_raw = create_4_candidate_mortages(tracks, ltv_bucket)
        labels = ["משכנתא נוכחית", "משכנתא מעודכנת", "משכנתא לא צמודה", "משכנתא מחזור אופטימלי"]
        
        comparison_table = []
        detailed_scenarios = {}
        total_loan_amount = sum(t['Payoff_Amount'] for t in tracks.values())

        # 3. ניתוח מעמיק של כל תרחיש ליצירת טבלאות וגרפים
        
        for i, label in enumerate(labels):
            scen_tracks_data = scenarios_raw[i]  # רשימת (info_dict, schedule)
            
            # עיבוד הנתונים המפורטים של התרחיש (לצילומי מסך 1 ו-6)
            scenario_detail = self._process_scenario_details(scen_tracks_data)
            detailed_scenarios[label] = scenario_detail
            
            # בניית שורה לטבלת ההשוואה (לצילום מסך 5)
            summary = scenario_detail["summary"]
            if label == "משכנתא נוכחית":
                first_month_payment_ori = summary["first_payment"]
                total_pay_ori = summary["total_repayment"]

            comparison_table.append({
                "תרחיש": label,
                "החזר חודשי ראשון": summary["first_payment"],
                "החזר חודשי מקסימלי": summary["max_payment"],
                "סך הכל תשלומים": summary["total_repayment"],
                "חיסכון ₪": 0 if i == 0 else detailed_scenarios[labels[0]]["summary"]["total_repayment"] - summary["total_repayment"],
                "החזר לשקל": summary["total_repayment"] / total_loan_amount if total_loan_amount > 0 else 0,
                "internal_rate_of_return":summary.get('internal_rate_of_return',0),
                "הפרש החזר חודשי ראשון": summary["first_payment"]- first_month_payment_ori,
                "חיסכון חודשי ממוצע": (detailed_scenarios[labels[0]]["summary"]["total_repayment"] - summary["total_repayment"])/len(scenario_detail["combined_graph"]["months"])
            })
            
        best_res_data = find_best_mortage(tracks, ltv_bucket)
        
        if best_res_data:
            # חישוב נתונים לגרף חיסכון בשנים (השוואה בין המצב הקיים לתמהיל הנבחר)
            # 1. שליפת זרם התשלומים של המצב הקיים (שכבר חושב)
            ori_payments = detailed_scenarios["משכנתא נוכחית"]["combined_graph"]["payments"]
            
            # 2. חישוב זרם התשלומים של התמהיל הנבחר
            best_details = self._process_scenario_details(best_res_data[0])
            best_payments = best_details["combined_graph"]["payments"]
            
            # 3. יישור לאותו אורך (Padding)
            max_len = max(len(ori_payments), len(best_payments))
            ori_padded = ori_payments + [0] * (max_len - len(ori_payments))
            best_padded = best_payments + [0] * (max_len - len(best_payments))
            
            years_x = []
            savings_y = []

            
            # 4. ביצוע החישוב השנתי (לא מצטבר)
            current_year_ori = 0
            current_year_best = 0
            
            for m in range(max_len):
                current_year_ori += ori_padded[m]
                current_year_best += best_padded[m]
                
                # דגימה בסוף כל שנה 
                if (m + 1) % 12 == 0:
                    years_x.append((m + 1) // 12)
                    savings_y.append(int(current_year_ori - current_year_best))
                    # איפוס לשנה הבאה
                    current_year_ori = 0
                    current_year_best = 0
            
            # הוספת נקודה אחרונה אם לא נפלנו בדיוק על שנה עגולה
            if max_len % 12 != 0:
                years_x.append(round(max_len / 12, 1))
                savings_y.append(int(current_year_ori - current_year_best))

            name_converter = {"update_mortage":"משכנתא מעודכנת","non_indx_mortage":"משכנתא לא צמודה","optimal_mortage":"משכנתא מחזור אופטימלי"}
            return {
                "comparison_table": comparison_table, # נתונים לטבלה ההשוואתית
                "detailed_scenarios": detailed_scenarios, # פירוט מלא לכל תרחיש (גרפים + טבלאות)
                "best_res": {'name': name_converter[best_res_data[4]], "גרף חיסכון בשנים": {"שנים": years_x, "חיסכון": savings_y}},
                "ltv_used": ltv_bucket
            }
        else:
            return {
                "comparison_table": comparison_table, # נתונים לטבלה ההשוואתית
                "detailed_scenarios": detailed_scenarios, # פירוט מלא לכל תרחיש (גרפים + טבלאות)
                "best_res": {'name': 'there is no saving!!'},
                "ltv_used": ltv_bucket
            }

    def _process_scenario_details(self, tracks_list: List) -> Dict[str, Any]:
        """
        מעבד רשימת מסלולים לנתונים מסכמים וגרפיים (זרם תשלומים, חלוקה שנתית).
        """
        
        
        track_list_detailed = []
        all_schedules = []
        total_p, total_i, total_k = 0, 0, 0
        
        for info, sch in tracks_list:
            all_schedules.append(sch)
            p, i, k = summarize_schedule(sch)
            total_p += p; total_i += i; total_k += k
            
            # נתונים למסלול בודד (לטבלה בצילום מסך 6)
            track_list_detailed.append({
                "מסלול": info.get("מסלול"),
                "סכום": info.get("סכום"),
                "תקופה_חודשים": info.get("תקופה (חודשים)"),#"תקופה (חודשים)"
                "ריבית": info.get("ריבית"),
                "החזר_ראשון": sch[0][2],
                "החזר_בשיא": max(row[2] for row in sch),
                "סהכ_ריבית_והצמדה": i + k,
                "סך_כל_התשלומים": p + i + k,
                'internal_rate_of_return': info.get('internal_rate_of_return',0),
                "graph_arrays": _schedule_arrays(sch) # כל המערכים לגרפים של מסלול בודד
            })

        # חישוב זרם תשלומים מאוחד לתרחיש (לגרף הכחול המצטבר)
        
        months, total_pmts = _aggregate_monthly_payment([t[1] for t in tracks_list])
        
        return {
            "summary": {
                "total_principal": total_p, # יתרה לסילוק (צילום 6)
                "total_interest": total_i,
                "total_indexation": total_k,
                "total_repayment": total_p + total_i + total_k,
                "first_payment": total_pmts[0],
                "max_payment": max(total_pmts)
            },
            "tracks": track_list_detailed,
            "combined_graph": {
                "months": months,
                "payments": total_pmts
            }
        }

    def check_approval_analysis(self, file_content: bytes, property_value: float) -> Dict[str, Any]:
        """
        ניתוח אישור עקרוני (טאב 6).
        משווה בין תמהיל מהקובץ לתמהיל אופטימלי ומחזיר את כל המידע הוויזואלי.
        """
        # 1. עיבוד מסלולים מהקובץ (תמהיל מוצע מהקובץ)
        try:
            api_json = json.loads(file_content.decode("utf-8-sig"))
            first_loan_tracks = convert_api_json_to_first_loan_tracks(api_json)
        except Exception as e:
            return {"error": f"Invalid JSON: {str(e)}"}

        original_tracks_data = []
        original_principal = 0
        monthly_payment_orig_first = 0
        max_months_orig = 0
        original_table = []
        for tr in first_loan_tracks:
            # לוגיקת בחירת תדירות וריבית פריים כפי שמופיעה ב-Functions
            freq_val = tr.get("freq")
            prime_offset = 0
            if tr["rate_type"] in ("מלצ", "מצ", 'מטח דולר', 'מטח יורו'):
                freq_val = int(freq_val) if freq_val else 60
            elif tr["rate_type"] == "פריים":
                freq_val = 1
                prime_offset = tr["rate"] - (con.bank_of_israel_rate + con.prime_margin)
            
            sch = calculate_schedule(
                tr["principal"], tr["months"], tr["rate"], 
                tr["schedule_t"], tr["rate_type"], freq_val, 
                con.prime_margin, prime_offset
            )
            
            original_tracks_data.append({"info": tr, "schedule": sch})
            original_principal += tr["principal"]
            monthly_payment_orig_first += sch[0][2]
            max_months_orig = max(max_months_orig, tr["months"])
            original_table.append({
                    "סוג_מסלול": tr["rate_type"],
                    "סכום": tr["principal"],
                    "תקופה_חודשים": tr["months"],
                    "ריבית": tr["rate"],
                    "החזר_חודשי": sch[0][2]
                })

        # 2. הרצת אופטימיזציה להשוואה (הסל האופטימלי)
        ltv_details = self.calculate_ltv_details(property_value, original_principal)
        
        # תיקון: שליפת הסטרינג מתוך המילון
        ltv_input_str = ltv_details["allocation"] 

        opt_sol, opt_totals, opt_err = optimize_mortgage(
            float(original_principal),
            ltv_input_str, 
            monthly_payment_orig_first * con.monthly_income_factor,
            con.sensitivity,
            con.prepay_window_key,
            con.durations_months(max_months_orig),
            con.bank_of_israel_rate,
            con.prime_margin,
            con.objective_mode,
            con.alpha,
            monthly_payment_orig_first,
        )

        # 3. פונקציית עזר לסיכום נתוני סנריו (המטריקות המופיעות בראש כל בלוק בצילום)
        def get_scenario_metrics(tracks_list):
            schedules = [t["schedule"] if isinstance(t, dict) else t["schedule"] for t in tracks_list]
            total_p, total_i, total_k = 0, 0, 0
            max_m = 0
            first_pmt = 0
            
            for sch in schedules:
                p, i, k = summarize_schedule(sch)
                total_p += p; total_i += i; total_k += k
                max_m = max(max_m, len(sch))
                first_pmt += sch[0][2]
                
            # חישוב החזר מקסימלי (שיא ההחזר מכל המסלולים יחד)
            max_pmt = 0
            for m_idx in range(max_m):
                current_month_total = sum(sch[m_idx][2] for sch in schedules if m_idx < len(sch))
                max_pmt = max(max_pmt, current_month_total)
                
            return {
                "סכום_הלוואה": total_p,
                "תקופה_מקסימלית": f"{max_m // 12} שנים ({max_m} חודשים)",
                "סהכ_החזר_כולל": total_p + total_i + total_k,
                "החזר_חודשי_ראשון": first_pmt,
                "החזר_חודשי_מקסימלי": max_pmt,
                "delta_pmt": max_pmt - first_pmt
            }

        # עיבוד פירוט מסלולים לסל האופטימלי (לטבלה בצילום)
        opt_table = []
        if not opt_err:
            for tr in opt_sol:
                opt_table.append({
                    "סוג_מסלול": tr["rate_type"],
                    "סכום": tr["principal"],
                    "תקופה_חודשים": tr["months"],
                    "ריבית": tr["rate"],
                    "החזר_חודשי": tr["schedule"][0][2]
                })

        # 4. מבנה נתונים סופי התואם לצילום המסך
        results = {
            "proposed_mix": {
                "metrics": get_scenario_metrics(original_tracks_data),
                "table": original_table,
                "label": "תמהיל מוצע (מהקובץ)"
            },
            "optimal_mix": {
                "metrics": get_scenario_metrics([{"schedule": t["schedule"]} for t in opt_sol]) if not opt_err else None,
                "table": opt_table,
                "label": "הסל האופטימלי",
                "error": opt_err
            },
            "savings": {
                "total_savings": 0,
                "best_name": "הסל האופטימלי"
            }
        }

        # תיקון קריטי: חישוב החיסכון רק אם יש מטריקות תקינות
        if results["optimal_mix"]["metrics"]:
            results["savings"]["total_savings"] = results["proposed_mix"]["metrics"]["סהכ_החזר_כולל"] - results["optimal_mix"]["metrics"]["סהכ_החזר_כולל"]
        else:
            results["savings"]["total_savings"] = 0

        return results
    
    def check_approval_analysis_new_format(self, file_content: bytes, property_value: float) -> Dict[str, Any]:
        """
        ניתוח אישור עקרוני (טאב 6).
        משווה בין תמהיל מהקובץ לתמהיל אופטימלי ומחזיר את כל המידע הוויזואלי.
        """
        # 1. עיבוד מסלולים מהקובץ (תמהיל מוצע מהקובץ)
        try:
            api_json = json.loads(file_content.decode("utf-8-sig"))
            first_loan_tracks = convert_api_json_to_first_loan_tracks(api_json)
        except Exception as e:
            return {"error": f"Invalid JSON: {str(e)}"}

        original_tracks_data = []
        original_principal = 0
        monthly_payment_orig_first = 0
        max_months_orig = 0
        total_pay_orig, total_int_orig, total_idx_orig, first_pmt_orig = 0, 0, 0, 0
        track_details_orig = []
        results = {}
        
        for tr in first_loan_tracks:
            # לוגיקת בחירת תדירות וריבית פריים כפי שמופיעה ב-Functions
            freq_val = tr.get("freq")
            prime_offset = 0
            if tr["rate_type"] in ("מלצ", "מצ", 'מטח דולר', 'מטח יורו'):
                freq_val = int(freq_val) if freq_val else 60
            elif tr["rate_type"] == "פריים":
                freq_val = 1
                prime_offset = tr["rate"] - (con.bank_of_israel_rate + con.prime_margin)
            
            sch = calculate_schedule(
                tr["principal"], tr["months"], tr["rate"], 
                tr["schedule_t"], tr["rate_type"], freq_val, 
                con.prime_margin, prime_offset
            )
            
            original_tracks_data.append({"info": tr, "schedule": sch})
            original_principal += tr["principal"]
            monthly_payment_orig_first += sch[0][2]
            max_months_orig = max(max_months_orig, tr["months"])
            
            p, i, k = summarize_schedule(sch)
            total_pay_orig += (p + i + k)
            total_int_orig += i
            total_idx_orig += k
            first_pmt_orig += sch[0][2]
            
            track_details_orig.append({
                "שם": tr["rate_type"],
                "ריבית": tr["rate"],
               "תקופה_חודשים": tr["months"],
                "החזר_חודשי": sch[0][2]
            })
        
        months_axis = list(range(1, max_months_orig + 1))
        principal_flow = [sum(t["schedule"][m-1][5] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]
        interest_flow = [sum(t["schedule"][m-1][3] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]
        indexation_flow = [sum(t["schedule"][m-1][6] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]

        results["proposed_mix"] = {
            "summary": {
                "סכום_הלוואה": original_principal,
                "סהכ_החזר_משוער": total_pay_orig,
                "מזה_הצמדה_למדד": total_idx_orig,
                "החזר_חודשי_ראשון": first_pmt_orig,
            },
            "tracks_detail": track_details_orig,
            
            "graph_data": {
                "months": months_axis,
                "principal_repayment": principal_flow,
                "interest_payment": interest_flow,
                "indexation_component": indexation_flow
            }
        }

        # 2. הרצת אופטימיזציה להשוואה (הסל האופטימלי)
        ltv_details = self.calculate_ltv_details(property_value, original_principal)
        
        # תיקון: שליפת הסטרינג מתוך המילון
        ltv_input_str = ltv_details["allocation"] 

        opt_sol, opt_totals, opt_err = optimize_mortgage(
            float(original_principal),
            ltv_input_str, 
            monthly_payment_orig_first * con.monthly_income_factor,
            con.sensitivity,
            con.prepay_window_key,
            con.durations_months(max_months_orig),
            con.bank_of_israel_rate,
            con.prime_margin,
            con.objective_mode,
            con.alpha,
            monthly_payment_orig_first,
        )

        total_pay_opt, total_int_opt, total_idx_opt, first_pmt_opt = 0, 0, 0, 0
        track_details_opt = []     
        max_months_opt = 0 
        if not opt_err:
            for tr in opt_sol:
                track_details_opt.append({
                "שם": tr["rate_type"],
                "ריבית": tr["rate"],
               "תקופה_חודשים": tr["months"],
                "החזר_חודשי": tr["schedule"][0][2]
                })

                p, i, k = summarize_schedule(tr["schedule"])
                total_pay_opt += (p + i + k)
                total_int_opt += i
                total_idx_opt += k
                first_pmt_opt += tr["schedule"][0][2]
                max_months_opt = max(max_months_opt, tr["months"])

        
            months_axis = list(range(1, max_months_opt + 1))
            principal_flow = [sum(t["schedule"][m-1][5] for t in opt_sol if m <= len(t["schedule"])) for m in months_axis]
            interest_flow = [sum(t["schedule"][m-1][3] for t in opt_sol if m <= len(t["schedule"])) for m in months_axis]
            indexation_flow = [sum(t["schedule"][m-1][6] for t in opt_sol if m <= len(t["schedule"])) for m in months_axis]

            results["optimal_mix"] = {
                "summary": {
                    "סכום_הלוואה": original_principal,
                    "סהכ_החזר_משוער": total_pay_opt,
                    "מזה_הצמדה_למדד": total_idx_opt,
                    "החזר_חודשי_ראשון": first_pmt_opt,
                },
                "tracks_detail": track_details_opt,
                
                "graph_data": {
                    "months": months_axis,
                    "principal_repayment": principal_flow,
                    "interest_payment": interest_flow,
                    "indexation_component": indexation_flow
                }
            }
                    
            savings = results["proposed_mix"]["summary"]["סהכ_החזר_משוער"] - results["optimal_mix"]["summary"]["סהכ_החזר_משוער"]
            if savings > 0:
                results["savings"] = {
                    "total_savings": savings,
                    "savings_percentage": savings / results["proposed_mix"]["summary"]["סהכ_החזר_משוער"] * 100
                }
            else:
                print("not saving!!!!",savings)
                results["savings"] = {
                    "total_savings": 0,
                    "savings_percentage": 0
                }

        else:
            results["optimal_mix"] = None
            results["savings"] = None
        return results

    def get_uniform_baskets_analysis(self, principal: float, years: int) -> Dict[str, Any]:
        """
        ניתוח סלים אחידים (טאב 7).
        מחשב 3 סלים מובנים ומחזיר את כל המידע המופיע בצילום המסך.
        """
        months = years * 12

        # פונקציית עזר לחישוב מסלול מהיר בתוך הסל
        def quick_calc(amt, t_type, freq, name):
            try:
                # משיכת ריבית מותאמת לפי תקופה ו-LTV (75% כברירת מחדל לסימולטור)
                rate = float(self.calculator.get_adjusted_rate(
                    con.ANCHOR_TRACK_MAP[t_type], '75%', freq, months
                ))
            except: 
                rate = 4.0
            
            sch = calculate_schedule(amt, months, rate, "שפיצר", t_type, freq, con.prime_margin, 0)
            return {"name": name, "sch": sch, "rate": rate, "principal": amt}

        # הגדרת הרכב שלושת הסלים
        baskets_config = {
            "סל אחיד 1": [
                quick_calc(principal, "קלצ", None, "קלצ מלא")
            ],
            "סל אחיד 2": [
                quick_calc(principal/3, "קלצ", None, "קלצ (1/3)"),
                quick_calc(principal/3, "פריים", 1, "פריים (1/3)"),
                quick_calc(principal/3, "מצ", 60, "משתנה (1/3)")
            ],
            "סל אחיד 3": [
                quick_calc(principal/2, "קלצ", None, "קלצ (1/2)"),
                quick_calc(principal/2, "פריים", 1, "פריים (1/2)")
            ]
        }

        results = {}

        for basket_name, tracks in baskets_config.items():
            total_pay, total_int, total_idx, first_pmt = 0, 0, 0, 0
            track_details = []
            
            # 1. עיבוד נתונים מסכמים ופירוט מסלולים (הקוביות הירוקות והבר הכתום בצילום)
            for t in tracks:
                p, i, k = summarize_schedule(t["sch"])
                total_pay += (p + i + k)
                total_int += i
                total_idx += k
                first_pmt += t["sch"][0][2]
                
                track_details.append({
                    "שם": t["name"],
                    "ריבית": t["rate"],
                    "סכום": t["principal"]
                })

            # 2. בניית נתונים לגרף "התפתחות החזרים לאורך זמן" (תחתית הצילום)
            months_axis = list(range(1, months + 1))
            principal_flow = [sum(t["sch"][m-1][5] for t in tracks if m <= len(t["sch"])) for m in months_axis]
            interest_flow = [sum(t["sch"][m-1][3] for t in tracks if m <= len(t["sch"])) for m in months_axis]
            indexation_flow = [sum(t["sch"][m-1][6] for t in tracks if m <= len(t["sch"])) for m in months_axis]

            results[basket_name] = {
                "summary": {
                    "סכום_הלוואה": principal,
                    "סהכ_החזר_משוער": total_pay,
                    "מזה_הצמדה_למדד": total_idx,
                    "החזר_חודשי_ראשון": first_pmt
                },
                "tracks_detail": track_details,
                "graph_data": {
                    "months": months_axis,
                    "principal_repayment": principal_flow,
                    "interest_payment": interest_flow,
                    "indexation_component": indexation_flow
                }
            }

        return results
    
engine = MortgageEngine()


@app.post("/config/sync")
async def sync_config(
    payload: ConfigSyncRequest,
    x_config_sync_key: Optional[str] = Header(default=None, alias="X-Config-Sync-Key"),
):
    _validate_sync_key(x_config_sync_key)
    if not payload.updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    config_path = Path(__file__).parent.resolve() / "config.py"
    _apply_config_updates(config_path, payload.updates)
    engine.refresh_runtime_config()
    return {
        "ok": True,
        "updated_keys": sorted(str(k) for k in payload.updates.keys()),
    }

@app.post("/simulate") 
async def simulate(data: List[Dict], property_value: float): # הסרנו את = Form(...)
    engine.refresh_runtime_config()
    return engine.simulate_manual_tracks(data, property_value)

@app.post("/optimize") # טאב 2
async def optimize(loan: float = Form(...), property_value: float = Form(...), income: float = Form(...), max_pmt: float = Form(...),max_scan_years: float = Form(...)):
    """
    נקודת קצה לביצוע אופטימיזציה למשכנתא חדשה (טאב 2).
    מקבלת את פרטי ההלוואה והנכס, ומחזירה תמהיל אופטימלי הכולל רשימת מסלולים,
    לוחות סילוקין, וניתוח החזרים.
    """
    engine.refresh_runtime_config()
    params = {"loan_amount": loan, "property_value": property_value, "income": income, "max_pmt": max_pmt, "max_scan_years": max_scan_years, "sensitivity": "בינוני", "mode": "balanced", "alpha": 0.5}
    return engine.run_optimization(params)

@app.post("/refinance") # טאב 3
async def refinance(json_file: UploadFile = File(...), property_value: float = Form(...)):
    engine.refresh_runtime_config()
    return engine.analyze_refinance(await json_file.read(),property_value)

@app.post("/approval-check") 
async def approval(json_file: UploadFile = File(...), property_value: float = Form(...)):
    engine.refresh_runtime_config()
    return engine.check_approval_analysis_new_format(await json_file.read(), property_value)

@app.get("/uniform-baskets") # טאב 7
async def baskets(principal: float, years: int):
    engine.refresh_runtime_config()
    return engine.get_uniform_baskets_analysis(principal, years)
