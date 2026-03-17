from fastapi import FastAPI, UploadFile, File, Form
import json
from typing import List, Dict, Any, Optional
from functions import (
    calculate_schedule, summarize_schedule, optimize_mortgage, 
    convert_api_json_to_loan_tracks,
    convert_api_json_to_first_loan_tracks, InterestRateCalculator,
    _schedule_arrays,_aggregate_monthly_payment,aggregate_yearly,
    find_best_mortage
)
import config as con
from fastapi.responses import JSONResponse


class RoundingJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        # פונקציית עזר שסורקת את הנתונים ומעגלת float
        def round_floats(obj):
            if isinstance(obj, float):
                return round(obj, 2)
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [round_floats(x) for x in obj]
            return obj

        return json.dumps(
            round_floats(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
# uvicorn api_server:app --reload
#app = FastAPI(title="Mortgage API Server")
app = FastAPI(title="Mortgage API Server", default_response_class=RoundingJSONResponse)
class MortgageEngine:
    def __init__(self):
        self.calculator = InterestRateCalculator()
    
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
        #loan_amount = sum([tracks[i]['Payoff_Amount'] for i in tracks.keys()])
        
        loan_amount = sum([tracks[i]['loan_value'] + tracks[i]['loan_value_inflation'] + tracks[i]['loan_value_interest'] + tracks[i]['fee_differences']  for i in tracks.keys()])
        ltv_bucket = self.calculate_ltv_details(property_value,loan_amount)['allocation']
        
        # 2. יצירת 4 תרחישי המחזור (מקור, מעודכנת, לא צמודה, אופטימלית)
        #scenarios_raw = create_4_candidate_mortages(tracks, ltv_bucket)
        labels = ["Current_Mortgage", "Updated_Mortgage", "Non-linked_Mortgage", "Optimal_Refinance_Mortgage"]
        best_res_data,ori_routes_mortage,ori_routes_mortage2, update_mortage, non_indx_mortage, optimal_mortage = find_best_mortage(tracks, ltv_bucket)
        scenarios_raw = [ori_routes_mortage, update_mortage, non_indx_mortage, optimal_mortage]
        comparison_table = []
        detailed_scenarios = {}
        total_loan_amount = loan_amount#sum(t['Payoff_Amount'] for t in tracks.values())

        # 3. ניתוח מעמיק של כל תרחיש ליצירת טבלאות וגרפים
        
        for i, label in enumerate(labels):
            if label == "Best_Mortage":
                continue
            scen_tracks_data = scenarios_raw[i]  # רשימת (info_dict, schedule)
            
            # עיבוד הנתונים המפורטים של התרחיש (לצילומי מסך 1 ו-6)
            scenario_detail = self._process_scenario_details(scen_tracks_data)
            detailed_scenarios[label] = scenario_detail
            
            # בניית שורה לטבלת ההשוואה (לצילום מסך 5)
            summary = scenario_detail["summary"]
            if label == "Current_Mortgage": # originally "משכנתא נוכחית"
                first_month_payment_ori = summary["first_payment"]
                total_pay_ori = summary["total_repayment"]
            tracks = scenario_detail["tracks"]
            len_tracks = len(tracks)
            internal_rate_of_return = 0 
            for track in tracks:
                internal_rate_of_return+=track["internal_rate_of_return"]

            internal_rate_of_return = internal_rate_of_return/ len_tracks   
            comparison_table.append({
                "Scenario": label, # originally "תרחיש": label
                "First_Monthly_Payment": summary["first_payment"], # originally "החזר חודשי ראשון": summary["first_payment"]
                "Max_Monthly_Payment": summary["max_payment"], # originally "החזר חודשי מקסימלי": summary["max_payment"]
                "Total_Repayment": summary["total_repayment"], # originally "סך הכל תשלומים": summary["total_repayment"]
                "Savings_NIS": 0 if i == 0 else detailed_scenarios[labels[0]]["summary"]["total_repayment"]-summary["total_repayment"] , # originally "חיסכון ₪"
                "Return_per_NIS": summary["total_repayment"] / total_loan_amount if total_loan_amount > 0 else 0, # originally "החזר לשקל"
                "internal_rate_of_return":internal_rate_of_return,#summary.get('internal_rate_of_return',0),
                "First_Monthly_Payment_Diff": summary["first_payment"]- first_month_payment_ori, # originally "הפרש החזר חודשי ראשון"
                "Average_Monthly_Savings": (detailed_scenarios[labels[0]]["summary"]["total_repayment"]-summary["total_repayment"])/len(scenario_detail["combined_graph"]["months"]), # originally "חיסכון חודשי ממוצע"
                "Saving_in_precentage":0 if i == 0 else (detailed_scenarios[labels[0]]["summary"]["total_repayment"]-summary["total_repayment"] )*100/detailed_scenarios[labels[0]]["summary"]["total_repayment"] ,
            })
            print("__________________________________")
            print(comparison_table[-1]['Saving_in_precentage'])
            
        
        
        if best_res_data:
            # חישוב נתונים לגרף חיסכון בשנים (השוואה בין המצב הקיים לתמהיל הנבחר)
            # 1. שליפת זרם התשלומים של המצב הקיים (שכבר חושב)
            ori_payments = detailed_scenarios["Current_Mortgage"]["combined_graph"]["payments"]  # originally "משכנתא נוכחית"
            
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

            # name_converter = {"update_mortage":"משכנתא מעודכנת","non_indx_mortage":"משכנתא לא צמודה","optimal_mortage":"משכנתא מחזור אופטימלי"}
            name_converter = {"update_mortage":"Updated_Mortgage","non_indx_mortage":"Non-linked_Mortgage","optimal_mortage":"Optimal_Refinance_Mortgage"}
            return {
                "comparison_table": comparison_table, # נתונים לטבלה ההשוואתית
                "detailed_scenarios": detailed_scenarios, # פירוט מלא לכל תרחיש (גרפים + טבלאות)
                "best_res": {'name': name_converter[best_res_data[4]], "Savings_Graph_By_Years": {"Years": years_x, "Savings": savings_y}}, # originally "גרף חיסכון בשנים": {"שנים": years_x, "חיסכון": savings_y}
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
                "Track": info.get("מסלול"), # originally "מסלול": info.get("מסלול")
                "Amount": info.get("סכום"), # originally "סכום": info.get("סכום")
                "Term_Months": info.get("תקופה (חודשים)"), # originally "תקופה_חודשים": info.get("תקופה (חודשים)")
                "Interest": info.get("ריבית"), # originally "ריבית": info.get("ריבית")
                "First_Payment": sch[0][2], # originally "החזר_ראשון": sch[0][2]
                "Max_Payment": max(row[2] for row in sch), # originally "החזר_בשיא": max(row[2] for row in sch)
                "Total_Interest": i , # originally "סהכ_ריבית_והצמדה": i + k
                "Total_Indexation": k ,
                "Total_Repayment": p + i + k, # originally "סך_כל_התשלומים": p + i + k
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
                "Name": tr["rate_type"], # originally "שם": tr["rate_type"]
                "Interest": tr["rate"], # originally "ריבית": tr["rate"]
                "Term_Months": tr["months"], # originally "תקופה_חודשים": tr["months"]
                "Monthly_Payment": sch[0][2] # originally "החזר_חודשי": sch[0][2]
            })
        
        months_axis = list(range(1, max_months_orig + 1))
        principal_flow = [sum(t["schedule"][m-1][5] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]
        interest_flow = [sum(t["schedule"][m-1][3] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]
        indexation_flow = [sum(t["schedule"][m-1][6] for t in original_tracks_data if m <= len(t["schedule"])) for m in months_axis]

        results["proposed_mix"] = {
            "summary": {
                "Loan_Amount": original_principal, # originally "סכום_הלוואה": original_principal
                "Total_Estimated_Repayment": total_pay_orig, # originally "סהכ_החזר_משוער": total_pay_orig
                "Indexation_Component": total_idx_orig, # originally "מזה_הצמדה_למדד": total_idx_orig
                "First_Monthly_Payment": first_pmt_orig, # originally "החזר_חודשי_ראשון": first_pmt_orig
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
                "Name": tr["rate_type"], # originally "שם": tr["rate_type"]
                "Interest": tr["rate"], # originally "ריבית": tr["rate"]
               "Term_Months": tr["months"], # originally "תקופה_חודשים": tr["months"]
                "Monthly_Payment": tr["schedule"][0][2] # originally "החזר_חודשי": tr["schedule"][0][2]
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
                    "Loan_Amount": original_principal, # originally "סכום_הלוואה": original_principal
                    "Total_Estimated_Repayment": total_pay_opt, # originally "סהכ_החזר_משוער": total_pay_opt
                    "Indexation_Component": total_idx_opt, # originally "מזה_הצמדה_למדד": total_idx_opt
                    "First_Monthly_Payment": first_pmt_opt, # originally "החזר_חודשי_ראשון": first_pmt_opt
                },
                "tracks_detail": track_details_opt,
                
                "graph_data": {
                    "months": months_axis,
                    "principal_repayment": principal_flow,
                    "interest_payment": interest_flow,
                    "indexation_component": indexation_flow
                }
            }
                    
            savings = results["proposed_mix"]["summary"]["Total_Estimated_Repayment"] - results["optimal_mix"]["summary"]["Total_Estimated_Repayment"] # originally "סהכ_החזר_משוער"
            if savings > 0:
                results["savings"] = {
                    "total_savings": savings,
                    "savings_percentage": savings / results["proposed_mix"]["summary"]["Total_Estimated_Repayment"] * 100 # originally "סהכ_החזר_משוער"
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
            "Uniform_Basket_1": [ 
                quick_calc(principal, "קלצ", None, "100% קל״צ") # originally "קלצ מלא"
            ],
            "Uniform_Basket_2": [ # originally "סל אחיד 2"
                quick_calc(principal/3, "קלצ", None, "33.3% קל״צ"), # originally "קלצ (1/3)"
                quick_calc(principal/3, "פריים", 1, "33.3% פריים"), # originally "פריים (1/3)"
                quick_calc(principal/3, "מצ", 60, "33.3 מ״צ") # originally "משתנה (1/3)"
            ],
            "Uniform_Basket_3": [ # originally "סל אחיד 3"
                quick_calc(principal/2, "קלצ", None, "50% קל״צ"), # originally "קלצ (1/2)"
                quick_calc(principal/2, "פריים", 1, "50% פריים") # originally "פריים (1/2)"
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
                    "Name": t["name"], # originally "שם": t["name"]
                    "Interest": t["rate"], # originally "ריבית": t["rate"]
                    "Amount": t["principal"] # originally "סכום": t["principal"]
                })

            # 2. בניית נתונים לגרף "התפתחות החזרים לאורך זמן" (תחתית הצילום)
            months_axis = list(range(1, months + 1))
            principal_flow = [sum(t["sch"][m-1][5] for t in tracks if m <= len(t["sch"])) for m in months_axis]
            interest_flow = [sum(t["sch"][m-1][3] for t in tracks if m <= len(t["sch"])) for m in months_axis]
            indexation_flow = [sum(t["sch"][m-1][6] for t in tracks if m <= len(t["sch"])) for m in months_axis]

            results[basket_name] = {
                "summary": {
                    "Loan_Amount": principal, # originally "סכום_הלוואה": principal
                    "Total_Estimated_Repayment": total_pay, # originally "סהכ_החזר_משוער": total_pay
                    "Indexation_Component": total_idx, # originally "מזה_הצמדה_למדד": total_idx
                    "First_Monthly_Payment": first_pmt # originally "החזר_חודשי_ראשון": first_pmt
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

@app.post("/simulate") 
async def simulate(data: List[Dict], property_value: float): # הסרנו את = Form(...)
    return engine.simulate_manual_tracks(data, property_value)

@app.post("/optimize") # טאב 2
async def optimize(loan: float = Form(...), property_value: float = Form(...), income: float = Form(...), max_pmt: float = Form(...),max_scan_years: float = Form(...)):
    """
    נקודת קצה לביצוע אופטימיזציה למשכנתא חדשה (טאב 2).
    מקבלת את פרטי ההלוואה והנכס, ומחזירה תמהיל אופטימלי הכולל רשימת מסלולים,
    לוחות סילוקין, וניתוח החזרים.
    """
    params = {"loan_amount": loan, "property_value": property_value, "income": income, "max_pmt": max_pmt, "max_scan_years": max_scan_years, "sensitivity": "בינוני", "mode": "balanced", "alpha": 0.5}
    return engine.run_optimization(params)

@app.post("/refinance") # טאב 3
async def refinance(json_file: UploadFile = File(...), property_value: float = Form(...)):
    return engine.analyze_refinance(await json_file.read(),property_value)

@app.post("/approval-check") 
async def approval(json_file: UploadFile = File(...), property_value: float = Form(...)):
    return engine.check_approval_analysis_new_format(await json_file.read(), property_value)

@app.get("/uniform-baskets") # טאב 7
async def baskets(principal: float, years: int):
    return engine.get_uniform_baskets_analysis(principal, years)

