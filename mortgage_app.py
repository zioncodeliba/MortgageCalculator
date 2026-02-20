from __future__ import annotations

# streamlit run  /Users/user/Desktop/BI/final_10/mortgage_app.py

# ------------------------------ CONSTANTS & IMPORTS ------------------------------
import io
import re
import sys
import importlib
import os
print(f"🐍 Python Version: {sys.version}")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from typing import List
import config as con
import mortgage_data_loader as dl
import json
import requests
from functions import InterestRateCalculator,calculate_schedule,summarize_schedule,aggregate_yearly
from functions import convert_api_json_to_loan_tracks, optimize_mortgage,monthly_to_yearly, find_best_mortage
from functions import _schedule_arrays,_pad,_aggregate_monthly_payment, build_anchor,create_4_candidate_mortages,convert_api_json_to_first_loan_tracks
import pprint

@st.cache_data(ttl=86400)
def get_cached_data():
    path_model,path_nominal,path_real = dl.fetch_latest_boi_excels()
    store = dl.load_workbook_data(path_model, con.HORIZON, con.SCENARIO)
    print(f'Data loaded from: {path_model}')
    return path_model, store

XLSX_PATH, DATASTORE = get_cached_data()


# from data_loader_service import DATASTORE #, NOMINAL_ANCHOR, REAL_ANCHOR, MAKAM_ANCOR
@st.cache_data
def get_all_schedules(sim_p, sim_m):
    # פונקציית עזר פנימית לחישוב מסלול בודד
    def quick_calc(amt, t_type, freq, name, months):
        try:
            rate = float(clac_rate_int.get_adjusted_rate(con.ANCHOR_TRACK_MAP[t_type], '75%', freq, months))
        except: 
            rate = 4.0
        sch = calculate_schedule(amt, months, rate, "שפיצר", t_type, freq, con.prime_margin, 0)
        return {"name": name, "sch": sch, "rate": rate, "principal": amt}

    # חישוב כל הסלים מראש
    data = {
        "סל אחיד 1": [quick_calc(sim_p, "קלצ", None, "קל\"צ מלא", sim_m)],
        "סל אחיד 2": [
            quick_calc(sim_p/3, "קלצ", None, "קל\"צ (1/3)", sim_m),
            quick_calc(sim_p/3, "פריים", 1, "פריים (1/3)", sim_m),
            quick_calc(sim_p/3, "מצ", 60, "משתנה (1/3)", sim_m)
        ],
        "סל אחיד 3": [
            quick_calc(sim_p/2, "קלצ", None, "קל\"צ (1/2)", sim_m),
            quick_calc(sim_p/2, "פריים", 1, "פריים (1/2)", sim_m)
        ]
    }
    return data

def save_config_to_file(updates: dict):
    """
    Reads config_v2.py, replaces variable assignments with new values,
    and writes it back. Supports both scalars and dictionaries.
    """
    #config_path = os.path.join(os.getcwd(),"config.py")
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "config.py"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        st.error(f"Failed to read config file: {e}")
        return

    for key, value in updates.items():
        if isinstance(value, (dict, list)):
            formatted_value = pprint.pformat(value, width=100, sort_dicts=False)
            
            # Simple prefix match for top-level assignment
            prefix_pattern = rf'(^|\n){key}\s*(?::[^=]+)?=\s*'
            
            # We assume unique variable names at top level
            p_match = re.search(prefix_pattern, content)
            if p_match:
                start_idx = p_match.end()
                # Find end of this structure (balanced braces)
                open_char = content[start_idx]
                close_char = '}' if open_char == '{' else ']'
                
                cnt = 0
                end_idx = start_idx
                in_struct = False
                
                # Scan forward
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
            # Handle scalars
            if isinstance(value, str):
                replacement = f'{key} = "{value}"'
                pattern = rf'{key}\s*=\s*[\'"][^\'"]*[\'"]'
            else:
                replacement = f'{key} = {value}'
                pattern = rf'{key}\s*=\s*[\d\.]+'

            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        st.error(f"Failed to write config file: {e}")


def sync_config_to_api(updates: dict):
    sync_url = os.getenv("CALC_API_SYNC_URL", "").strip()
    if not sync_url:
        return False, "CALC_API_SYNC_URL is not configured"

    headers = {"Content-Type": "application/json"}
    sync_key = os.getenv("CALC_API_SYNC_KEY", "").strip()
    if sync_key:
        headers["X-Config-Sync-Key"] = sync_key

    try:
        response = requests.post(
            sync_url,
            json={"updates": updates},
            headers=headers,
            timeout=20,
        )
    except requests.RequestException as e:
        return False, str(e)

    if not response.ok:
        try:
            error_payload = response.json()
            detail = error_payload.get("detail")
            if detail:
                return False, str(detail)
        except Exception:
            pass
        return False, f"HTTP {response.status_code}"

    return True, None

# ------------------------------ STREAMLIT UI (MIX) ------------------------------
st.set_page_config(page_title="מחשבון משכנתא", layout="wide")
st.title("מחשבון משכנתא")


clac_rate_int = InterestRateCalculator()
inflation_and_XLSX_PATH = XLSX_PATH
st.info(f"{inflation_and_XLSX_PATH} :נתוני אינפלציה וריבית נטענים מהקובץ הבא") #  
tab1, tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["📊 סימולטור מסלולים", "⚖️  תמהיל אופטימלי למשכנתא חדשה","תמהיל אופטימלי למחזור פנימי","מידע כללי", "הגדרות",'אישור עקרוני',"סימולטור לקוח"])

with tab1:
    # --- State init ---
    if "mix_tracks" not in st.session_state:
        st.session_state.mix_tracks = [
            {
                "principal": 1_000_000,
                "months": 240,
                "rate": 4.0,
                "ancor": 0,
                "spr":0,
                "loan_adj_global":0,
                "rate_type": "קלצ",
                "freq": 60,  # ערך ברירת מחדל ל'מלצ/מצ'
                "schedule_t": "שפיצר (החזר חודשי קבוע)",
                "spred": 0.0,
            }
        ]

    # בקובץ הראשי (mortgage_app_optimized-10.py) לפני הקריאה ל- render_optimizer_ui:
    tracks = st.session_state.get("mix_tracks", [])

    # ודא מזהה יציב לכל מסלול (UID) כדי למנוע התנגשויות מפתחות
    if "track_uid_counter" not in st.session_state:
        st.session_state.track_uid_counter = 0
    for tr in tracks:
        if "uid" not in tr:
            st.session_state.track_uid_counter += 1
            tr["uid"] = f"t{st.session_state.track_uid_counter}"

    # --- Tracks editor ---
    st.subheader("סימולטור מסלולים")
    add_col, _, _ = st.columns([1, 2, 2])
    if add_col.button("➕ הוסף מסלול", use_container_width=True):
        st.session_state.track_uid_counter += 1
        uid = f"t{st.session_state.track_uid_counter}"
        # Try to calc initial rate based on defaults
        init_months = 240
        init_type = "קלצ"
        init_ltv = st.session_state.get("capital_allocation", con.defult_capital_allocation)
        try:
            init_rate = clac_rate_int.get_adjusted_rate(
                con.ANCHOR_TRACK_MAP[init_type],
                init_ltv,
                60, # default freq
                init_months
            )
        except:
            init_rate = 4.0

        tracks.append({
            "uid": uid,
            "principal": 500_000,
            "months": init_months,
            "rate": float(init_rate),
            "ancor": 0,
            "spr":0,
            "loan_adj_global":0,
            "rate_type": init_type,
            "freq": 60,
            "schedule_t": "שפיצר (החזר חודשי קבוע)",
        })

    # --- Callbacks for Rate Updates ---
    def update_rate_cb(uid_target):
        # Retrieve current inputs from session state
        r_type = st.session_state.get(f"rate_type_{uid_target}")
        months = st.session_state.get(f"months_{uid_target}")
        freq = st.session_state.get(f"freq_{uid_target}") # Might be None for some types
        ltv = st.session_state.get("capital_allocation", con.defult_capital_allocation)
        
        # Safe defaults
        if not r_type: r_type = "קלצ"
        if not months: months = 240
        
        # Handle freq default if missing but needed
        # (Logic similar to main loop default)
        if r_type in ("מלצ", "מצ") and not freq: freq = 60
        
        try:
            new_r = clac_rate_int.get_adjusted_rate(
                con.ANCHOR_TRACK_MAP.get(r_type, "fixed_const_no_index"),
                ltv,
                freq,
                months
            )
            # Update the rate widget state
            st.session_state[f"rate_{uid_target}"] = float(new_r)
        except Exception as e:
            # print(f"Error in callback: {e}")
            pass

    def update_all_rates_cb():
        # Iterate all existing tracks in session and update them
        # Note: 'tracks' variable is local here, we need to access session state
        all_tr = st.session_state.get("mix_tracks", [])
        for t in all_tr:
            if "uid" in t:
                update_rate_cb(t["uid"])


    # בחירת הקצאת הון לכל התמהיל
    capital_options = ["35%", "50%", "60%", "75%", "100%" ,'any']
    capital_allocation = st.selectbox(
        "בחר הקצאת הון (LTV)",
        options=["35%", "50%", "60%", "75%", "100%" ,'any'],
        index=capital_options.index(con.defult_capital_allocation),  
        key="capital_allocation",
        on_change=update_all_rates_cb
    )
    #clac_rate_int = InterestRateCalculator()
    remove_index = None
    for i, tr in enumerate(tracks):
        with st.expander(f"מסלול #{i+1}", expanded=True):
            c1, c2, c3, c4, c5, c6, c7,c8 = st.columns(8)
            principal = c1.number_input("קרן (₪)", min_value=0, value=int(tr["principal"]), step=50_000, key=f"principal_{tr['uid']}")
            months    = c2.number_input("תקופה (חודשים)", min_value=1, value=int(tr["months"]), step=12, key=f"months_{tr['uid']}", on_change=update_rate_cb, args=(tr["uid"],))
            rate_type = c4.selectbox("סוג ריבית", ["קלצ", "קצ","מלצ","מצ", "פריים","מטח דולר","מטח יורו","מקמ","זכאות","מענק"], index=["קלצ","קצ","מלצ","מצ","פריים","מטח דולר","מטח יורו","מקמ","זכאות","מענק"].index(tr["rate_type"]), key=f"rate_type_{tr['uid']}", on_change=update_rate_cb, args=(tr["uid"],))

            # freq per type
            with c5:
                # ברירת מחדל בטוחה ל-FREQ אם דרוש
                prev_freq = tr.get("freq")
                safe_default_freq = int(prev_freq) if isinstance(prev_freq, (int, float)) and prev_freq else 60
                if rate_type in ("מלצ", "מצ"):
                    freq = st.number_input("תדירות שינוי ריבית (חודשים)", min_value=1, value=safe_default_freq, step=1, key=f"freq_{tr['uid']}", on_change=update_rate_cb, args=(tr["uid"],))
                elif rate_type == "מקמ":
                    freq = 12
                    st.caption("תדירות שינוי ריבית: שנתית (12)")
                elif rate_type == "פריים":
                    freq = 1
                    st.caption("תדירות שינוי ריבית: חודשית (1)")
                elif 'מטח' in rate_type:
                    freq = st.number_input("תדירות שינוי ריבית (חודשים)", min_value=1, value=1, step=1, key=f"freq_{tr['uid']}", on_change=update_rate_cb, args=(tr["uid"],))

                else:
                    freq = None

            try:  
                
                default_rate = clac_rate_int.get_adjusted_rate(
                    con.ANCHOR_TRACK_MAP[rate_type],
                    capital_allocation,
                    freq,
                    months, 
                )
                #print(f"default_rate: {default_rate}")

            except:
                # print('Fail to calc default_rate, use default_rate = 0')
                default_rate = 0
                #import pdb;pdb.set_trace()

            with c3:
                rate  = c3.number_input("ריבית שנתית (%)", min_value=0.0, value=float(default_rate), step=0.0001,format="%.12f", key=f"rate_{tr['uid']}")


            

            schedule_t = c6.selectbox("לוח סילוקין",
                                    ["שפיצר (החזר חודשי קבוע)",
                                    "קרן שווה (החזר חודשי יורד)",
                                    "בלון מלא", "בלון חלקי"],
                                    index=["שפיצר (החזר חודשי קבוע)","קרן שווה (החזר חודשי יורד)","בלון מלא","בלון חלקי"].index(tr["schedule_t"]),
                                    key=f"schedule_t_{tr['uid']}")
            
            
            # עמודת אחוז בתמהיל
            with c7:
                try:
                    total_principal_all = 0.0
                    for _t in tracks:
                        if _t.get("uid") == tr.get("uid"):
                            total_principal_all += float(principal)
                        else:
                            total_principal_all += float(_t.get("principal", 0) or 0)
                except Exception:
                    total_principal_all = 0.0
                mix_pct = (float(principal) / total_principal_all * 100.0) if total_principal_all > 0 else 0.0
                st.metric("אחוז בתמהיל", f"{mix_pct:.1f}%", f"key=pct_{tr['uid']}")

            # update state
            tr.update({
                "principal": principal,
                "months": months,
                "rate": rate,
                "rate_type": rate_type,
                "freq": freq,
                "schedule_t": schedule_t,
                
            })
            #print(f"default_rate: {default_rate}")
            if st.button("🗑️ הסר מסלול", key=f"remove_{tr['uid']}"):
                remove_index = i

    if remove_index is not None and 0 <= remove_index < len(tracks):
        del tracks[remove_index]

    # --- Compute button ---
    compute = st.button("חשב תמהיל", type="primary")


    if compute and len(tracks) > 0:
        # --- per-track schedules ---
        results: List[tuple[str, List[List[float]]]] = []
        max_len = 0
        errors = []
        for i, tr in enumerate(tracks):
            try:
                # נרמול תדירות לפי סוג המסלול לפני חישוב
                freq_val = tr.get("freq")
                prime_offset = 0
                if tr["rate_type"] in ("מלצ","מצ",'מטח דולר','מטח יורו'):
                    try:
                        freq_val = int(freq_val) if freq_val else 60
                    except Exception:
                        freq_val = 60
                elif tr["rate_type"] == "מקמ":
                    freq_val = 12
                elif tr["rate_type"] == "פריים":
                    freq_val = 1
                    # Offset = (Input Rate) - (Base Rate)
                    # Base Rate = BOI + Margin
                    current_base = con.bank_of_israel_rate + con.prime_margin
                    prime_offset = tr["rate"] - current_base

                else:
                    freq_val = None
                #print('tr["rate"]',tr["rate"])
                print('simulator in',tr["principal"], tr["months"], tr["rate"], tr["schedule_t"], tr["rate_type"], freq_val, con.prime_margin, prime_offset)
                sch = calculate_schedule(tr["principal"], tr["months"], tr["rate"], tr["schedule_t"], tr["rate_type"], freq_val,prime_margin= con.prime_margin,prime_offset= prime_offset)
                results.append((tr["uid"], sch))
                max_len = max(max_len, len(sch))
            except Exception as e:
                errors.append((i+1, str(e)))
        if errors:
            for idx, msg in errors:
                st.error(f"שגיאה במסלול #{idx}: {msg}")
            # st.stop() # Don't stop, just don't save results? Or stop? 
            # If errors, we probably shouldn't show partial results. 
            pass 
        else:
             st.session_state["manual_results"] = results
             st.rerun()

    # --- Render Logic (Persistent) ---
    results = st.session_state.get("manual_results")
    if results:
        # --- per-track tabs & plots ---
        tabs = st.tabs([f"מסלול #{i+1}" for i in range(len(results))])
        for i, (tab, (uid, sch)) in enumerate(zip(tabs, results)):
            with tab:
                xs, opening, payment, interest, principal_base, indexation, closing, used_rate = _schedule_arrays(sch)
                
                total_principal, total_interest, total_index = summarize_schedule(sch)
                st.markdown(f"**סכום קרן ששולמה (מסלול #{i+1}):** {total_principal:,.0f} ₪")
                st.markdown(f"**סה״כ ריבית ששולמה (מסלול #{i+1}):** {total_interest:,.0f} ₪")
                st.markdown(f"**סה״כ הצמדה ששולמה (מסלול #{i+1}):** {total_index:,.0f} ₪")
                st.markdown(f"**סך הכל תשלומים (מסלול #{i+1}):** {(total_principal+total_interest+total_index):,.0f} ₪")

                fig_open_debt = go.Figure()
                fig_open_debt.add_trace(go.Scatter(x=xs, y=opening, mode="lines", name="יתרה"))
                fig_open_debt.update_layout(title="יתרת פתיחת חוב לאורך הזמן", xaxis_title="חודש", yaxis_title="יתרת חוב (₪)")

                fig_pay = go.Figure()
                fig_pay.add_trace(go.Scatter(x=xs, y=payment, mode="lines", name="החזר חודשי"))
                fig_pay.update_layout(title="החזר חודשי לאורך הזמן", xaxis_title="חודש", yaxis_title="סכום (₪)")

                fig_close_debt = go.Figure()
                fig_close_debt.add_trace(go.Scatter(x=xs, y=closing, mode="lines", name="יתרה"))
                fig_close_debt.update_layout(title="יתרת סגירת חוב לאורך הזמן", xaxis_title="חודש", yaxis_title="יתרת חוב (₪)")

                fig_rate = go.Figure()
                fig_rate.add_trace(go.Scatter(x=xs, y=used_rate, mode="lines", name="ריבית אפקטיבית בפועל"))
                fig_rate.update_layout(title="התפתחות הריבית האפקטיבית לאורך התקופה", xaxis_title="חודש", yaxis_title="ריבית (%)")

                pr_y, in_y, idx_y = aggregate_yearly(sch)
                fig_br = go.Figure()
                fig_br.add_trace(go.Bar(x=list(range(1, len(pr_y) + 1)), y=pr_y,  name="קרן",marker=dict(color='blue')))
                fig_br.add_trace(go.Bar(x=list(range(1, len(idx_y) + 1)), y=idx_y, name="הצמדה",marker=dict(color='green')))
                fig_br.add_trace(go.Bar(x=list(range(1, len(in_y) + 1)), y=in_y,  name="ריבית",marker=dict(color='red')))
                fig_br.update_layout(title="חלוקת התשלום (שנתי)", barmode="stack", xaxis_title="שנה", yaxis_title="סכום (₪)")

                colA, colB ,colC,colD,colE= st.columns(5)
                with colA:
                    st.plotly_chart(fig_open_debt, width="stretch", key=f"track_{uid}_open")
                with colB:
                    st.plotly_chart(fig_close_debt, width="stretch", key=f"track_{uid}_close")
                with colC:
                    st.plotly_chart(fig_br, width="stretch", key=f"track_{uid}_br")
                with colD:
                    st.plotly_chart(fig_pay, width="stretch", key=f"track_{uid}_pay")
                with colE:
                    st.plotly_chart(fig_rate, width="stretch", key=f"track_{uid}_rate")

                

                # --- לוח סילוקין למסלול זה ---
                df_track = pd.DataFrame(
                    sch,
                    columns=[
                        "חודש","קרן תחילת תקופה","תשלום חודשי","ריבית",
                        "פירעון קרן (מוצמד)","פירעון קרן (בסיס)","רכיב הצמדה",
                        "יתרת סגירה","ריבית אפקטיבית שנתית (%)"
                    ]
                )
                try:
                    df_track["חודש"] = df_track["חודש"].astype(int)
                except Exception:
                    pass
                st.dataframe(
                    df_track.style.format({
                        "יתרת פתיחה": "₪{:,.0f}",
                        "תשלום חודשי": "₪{:,.0f}",
                        "תשלום ע''ח ריבית": "₪{:,.0f}",
                        "פירעון קרן (בסיס)": "₪{:,.0f}",
                        "פירעון קרן (מוצמד)": "₪{:,.0f}",
                        "רכיב הצמדה": "₪{:,.0f}",
                        "יתרת סגירה": "₪{:,.0f}",
                        "ריבית אפקטיבית שנתית (%)": "{:.2f}%"
                    }),
                    height=200, width="stretch", key=f"track_{uid}_table"
                )

                # הורדת CSV/Excel למסלול
                colA1,colB2 = st.columns(2)
                with colA1:
                    csv_bytes = df_track.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "הורד CSV (מסלול)", csv_bytes,
                        file_name=f"schedule_track_{i+1}.csv", mime="text/csv",
                        key=f"dl_csv_{uid}"
                    )
                with colB2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df_track.to_excel(writer, index=False, sheet_name=f"מסלול_{i+1}")
                    buf.seek(0)
                    st.download_button(
                        "הורד Excel (מסלול)", buf.getvalue(),
                        file_name=f"schedule_track_{i+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_xlsx_{uid}"
                    )

        # --- Combined series ---
        max_len = max([len(sch) for _, sch in results]) if results else 0
        L = max_len
        X = list(range(1, L + 1))
        total_opening = [0.0] * L
        total_closing = [0.0] * L
        total_payment = [0.0] * L
        total_interest = [0.0] * L
        total_principal_base = [0.0] * L
        total_indexation = [0.0] * L
        total_used_rate = [0.0] * L

        for _, sch in results:
            xs, opening, payment, interest, principal_base, indexation, closing, used_rate = _schedule_arrays(sch)
            total_opening   = [a + b for a, b in zip(total_opening, _pad(opening, L))]
            total_closing   = [a + b for a, b in zip(total_closing, _pad(closing, L))]
            total_payment   = [a + b for a, b in zip(total_payment, _pad(payment, L))]
            total_interest  = [a + b for a, b in zip(total_interest, _pad(interest, L))]
            total_principal_base = [a + b for a, b in zip(total_principal_base, _pad(principal_base, L))]
            total_indexation = [a + b for a, b in zip(total_indexation, _pad(indexation, L))]
            total_used_rate = [a + b for a, b in zip(total_used_rate, _pad(used_rate, L))]
        
        st.subheader("גרפים מסכמים (כל המסלולים)")
        # Combined totals
        total_P = 0.0
        total_I = 0.0
        total_K = 0.0
        for _, sch in results:
            P, I, K = summarize_schedule(sch)
            total_P += P; total_I += I; total_K += K
        st.markdown(f"**סכום קרן ששולמה – כל המסלולים:** {total_P:,.0f} ₪")
        st.markdown(f"**סה״כ ריבית ששולמה – כל המסלולים:** {total_I:,.0f} ₪")
        st.markdown(f"**סה״כ הצמדה ששולמה – כל המסלולים:** {total_K:,.0f} ₪")
        st.markdown(f"**סך כל התשלומים – כל המסלולים:** {(total_P + total_I + total_K):,.0f} ₪")
        
        
        # Combined plots
        fig_tot_open = go.Figure()
        fig_tot_open.add_trace(go.Scatter(x=X, y=total_opening, mode="lines", name="יתרה כוללת בתחילת חודש"))
        fig_tot_open.update_layout(title="יתרת פתיחת חוב – כל המסלולים", xaxis_title="חודש", yaxis_title="₪")

        fig_tot_close = go.Figure()
        fig_tot_close.add_trace(go.Scatter(x=X, y=total_closing, mode="lines", name="יתרה כוללת בסוף חודש"))
        fig_tot_close.update_layout(title="יתרת סגירת חוב – כל המסלולים", xaxis_title="חודש", yaxis_title="₪")

        fig_tot_pay = go.Figure()
        fig_tot_pay.add_trace(go.Scatter(x=X, y=total_payment, mode="lines", name="תשלום חודשי כולל"))
        fig_tot_pay.update_layout(title="החזר חודשי כולל (סכום כל המסלולים)", xaxis_title="חודש", yaxis_title="₪")

        # Yearly aggregate combined
        years = (L - 1) // 12 + 1
        y_idx = list(range(1, years + 1))
        pr_y  = [0.0] * years
        in_y  = [0.0] * years
        idx_y = [0.0] * years
        for m in range(1, L + 1):
            y = (m - 1) // 12
            pr_y[y]  += total_principal_base[m - 1]
            in_y[y]  += total_interest[m - 1]
            idx_y[y] += total_indexation[m - 1]

        fig_tot_br = go.Figure()
        fig_tot_br.add_trace(go.Bar(x=y_idx, y=pr_y,  name="קרן", marker=dict(color='blue')))
        fig_tot_br.add_trace(go.Bar(x=y_idx, y=idx_y, name="הצמדה", marker=dict(color='green')))
        fig_tot_br.add_trace(go.Bar(x=y_idx, y=in_y,  name="ריבית", marker=dict(color='red')))
        fig_tot_br.update_layout(title="חלוקת התשלום (שנתי) – כל המסלולים", barmode="stack", xaxis_title="שנה", yaxis_title="₪")
        
        fig_tot_used_rate = go.Figure()
        fig_tot_used_rate.add_trace(go.Scatter(x=X, y=total_used_rate, mode="lines", name="הריבית האפקטיבית"))
        fig_tot_used_rate.update_layout(title="הריבית האפקטיבית – כל המסלולים", xaxis_title="חודש", yaxis_title="₪")
        #total_used_rate

        colAll1,colAll2,colAll3,colAll4,colAll5 = st.columns(5)
        with colAll1:
            st.plotly_chart(fig_tot_open, width="stretch", key="total_open")
        with colAll2:
            st.plotly_chart(fig_tot_close, width="stretch", key="total_close")
        with colAll3:
            st.plotly_chart(fig_tot_pay, width="stretch", key="total_pay")
        with colAll4:
            st.plotly_chart(fig_tot_br, width="stretch", key="total_br")
        with colAll5:
            st.plotly_chart(fig_tot_used_rate, width="stretch", key="total_ur")
        

        # --- לוח סילוקין כולל (סכום כל המסלולים) ---
        principal_indexed_total = [a + b for a, b in zip(total_principal_base, total_indexation)]
        df_total = pd.DataFrame({
            "חודש": X,
            "קרן תחילת תקופה": total_opening,
            "תשלום חודשי": total_payment,
            "ריבית": total_interest,
            "פירעון קרן (מוצמד)": principal_indexed_total,
            "פירעון קרן (בסיס)": total_principal_base,
            "רכיב הצמדה": total_indexation,
            "יתרת סגירה": total_closing,
        })
        st.dataframe(
            df_total.style.format({
                "יתרת פתיחה": "₪{:,.0f}",
                "תשלום חודשי": "₪{:,.0f}",
                "ריבית": "₪{:,.0f}",
                "פירעון קרן (מוצמד)": "₪{:,.0f}",
                "פירעון קרן (בסיס)": "₪{:,.0f}",
                "רכיב הצמדה": "₪{:,.0f}",
                "יתרת סגירה": "₪{:,.0f}"
            }),
            height=200, width="stretch", key="total_table"
        )

        # הורדות לוח כולל
        csv_bytes_total = df_total.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "הורד CSV (תמהיל)", csv_bytes_total,
            file_name="schedule_total_mix.csv", mime="text/csv",
            key="dl_total_csv"
        )
        buf_total = io.BytesIO()
        with pd.ExcelWriter(buf_total, engine="openpyxl") as writer:
            df_total.to_excel(writer, index=False, sheet_name="תמהיל")
        buf_total.seek(0)
        st.download_button(
            "הורד Excel (תמהיל)", buf_total.getvalue(),
            file_name="schedule_total_mix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_total_xlsx"
        )

with tab2:
    st.subheader("תמהיל אופטימלי ללקיחת משכנתא חדשה")
    mix = st.session_state.get("mix_tracks", [])
    default_loan = sum(float(t.get("principal", 0) or 0.0) for t in mix) or 1_000_000.0
    c1, c2, c3 ,c4, c5= st.columns(5)
    loan_amount = c1.number_input("סכום הלוואה (₪)", min_value=1.0, value=float(default_loan), step=1.0, format="%.0f",key="loan_amount_input_tab2", help="הזן מספר ללא פסיקים. התצוגה בתוצאות תכלול פסיקים.")
    ltv_input = c2.selectbox("הקצאת הון (LTV)", ['35%','50%','60%','75%','100%','any'])
    monthly_income = c3.number_input("הכנסה חודשית נטו (₪)", min_value=1.0, value=20_000.0, step=1.0, format="%.0f", help="הזן מספר ללא פסיקים.")
    
    sensitivity = c4.selectbox("רגישות לריבית", ["נמוך", "בינוני", "גבוה"], index=0)
    max_monthly_payment = c5.number_input("מקסימום החזר חודשי(₪)", min_value=1.0, value=6_000.0, step=1.0, format="%.0f")

    c7a, c7b = st.columns([1, 1])

    objective_mode = c7a.selectbox(
        "מטרת האופטימיזציה",
        ["total_cost", "pmt1", "balanced"],
        index=0,
        format_func=lambda k: {"total_cost":"מזעור עלות כוללת", "pmt1":"מזעור החזר חודשי ראשון", "balanced":"מאוזן (עלות↔החזר)"}[k]
    )
    alpha = 0.5
    if objective_mode == "balanced":
        alpha = c7b.slider("α (משקל עלות)", 0.0, 1.0, 0.7, 0.05)
    
    c8, c9 = st.columns(2)
    prepay_window_key = c8.selectbox("חלון פירעון מוקדם חזוי", ["לא","כן","לא בטוח"], index=0)
    durations_sel = [] # Placeholder
    max_scan_years = c9.number_input("מקסימום שנים (עד)", min_value=12, max_value=40, value=30, step=1)
    
    # Logic requested: range(12, user_years + 1, 1) -> interpreted as Years -> converted to months
    durations_months = max_scan_years*12#[y * 12 for y in range(12, int(max_scan_years) + 1)]

    bank_rate_input = con.bank_of_israel_rate 
    prime_margin_input = con.prime_margin

    if st.button("🔎 מצא תמהיל אופטימלי", use_container_width=True):
        with st.spinner("מריץ אופטימיזציה..."): 
            sol_tracks, totals, err = optimize_mortgage(
                float(loan_amount),
                str(ltv_input),
                float(monthly_income),
                str(sensitivity),
                str(prepay_window_key),
                con.durations_months(max_scan_years*12),
                float(bank_rate_input),
                float(prime_margin_input),
                str(objective_mode),
                float(alpha),
                max_monthly_payment,
                None,#routes_data_ori = 
            )
            
        if err:
            st.error(err); 
            print(err)
        else:
            # Save to session (persistence)
            st.session_state["opt_solution"] = {"tracks": sol_tracks, "totals": totals}
            st.rerun()

    # --- Render Logic (Persistent) ---
    opt = st.session_state.get("opt_solution")
    if opt:
        sol_tracks = opt["tracks"]
        totals = opt["totals"]

        st.success("נמצא תמהיל אופטימלי (תוצאה שמורה)")
        if getattr(con, 'manual_interest_increase', 0) > 0:
            st.warning(f"⚠️ שימו לב: התוצאות כוללות תוספת ריבית ידנית של {con.manual_interest_increase}% (באופן יחסי).")
        
        # טבלה כולל אחוז מסך ההלוואה
        cols = ["תדירות שינוי","סוג ריבית", "תקופה (חודשים)", "ריבית שנתית (%)", "קרן (₪)", "אחוז מההלוואה (%)"]
        data, schedules = [], []
        for tr in sol_tracks:
            prc = float(tr["principal"])
            # Calc share based on current loan_amount input or stored? 
            # Ideally stored but for now using current input is approximation or we can store total loan.
            # But 'sol_tracks' has principals.
            total_principal_opt = sum(t["principal"] for t in sol_tracks)
            share = (prc / total_principal_opt) * 100.0 if total_principal_opt else 0.0
            data.append([tr['freq'],tr["rate_type"], int(tr["months"]), float(tr["rate"]), prc, round(share, 2)])
            schedules.append(tr["schedule"])
        
        df_mix = pd.DataFrame({
            cols[0]: [d[0] for d in data],
            cols[1]: [d[1] for d in data],
            cols[2]: [d[2] for d in data],
            cols[3]: [d[3] for d in data],
            cols[4]: [d[4] for d in data],
            cols[5]: [d[5] for d in data]
        })
        st.dataframe(
            df_mix.style.format({
                cols[3]: "{:.2f}%",
                cols[4]: "{:,.0f}",
                cols[5]: "{:.2f}%"
            }),
            width="stretch"
        )
        cA, cB, cC, cD = st.columns(4)
        cA.metric("סך הכל תשלומים", f"₪{totals['total_payment']:,.0f}")
        cB.metric("סה\"כ ריבית", f"₪{totals['total_interest']:,.0f}")
        cC.metric("סה\"כ הצמדה", f"₪{totals['total_indexation']:,.0f}")
        cD.metric("החזר חודשי ראשון משוער", f"₪{totals['pmt1']:,.0f}")
        
        months, total_pmts = _aggregate_monthly_payment(schedules)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=total_pmts, mode="lines", name="תשלום חודשי כולל"))
        fig.update_layout(title="זרם תשלומים חודשי (מצטבר מכל המסלולים)", xaxis_title="חודש", yaxis_title="₪")
        st.plotly_chart(fig, width="stretch")

        # --- Debug Visualization (Internal) ---
        # Rendered only if 'debug_data' is present in totals
        debug_data = totals.get("debug_data")
        if debug_data:
            st.divider()
            st.markdown("### 🔍 ניתוח מעמיק למנוע האופטימיזציה")
            
            candidates = debug_data.get("candidates", [])
            if candidates:
                df_debug = pd.DataFrame(candidates)
                
                # Numeric conversion
                cols_dbg = ["eff_cost", "base_norm", "pmt1_norm", "discount", "cost", "pmt1", "months"]
                for c in cols_dbg:
                     if c in df_debug.columns:
                        df_debug[c] = pd.to_numeric(df_debug[c], errors='coerce').fillna(0)
                        
                # Display tweaks
                df_debug["Years"] = df_debug["months"] / 12.0
                df_debug["Size"] = df_debug["selected"].apply(lambda x: 15 if x else 8)
                
                d1, d2 = st.columns(2)
                
                # Chart 1
                with d1:
                    fig1 = px.scatter(
                        df_debug, x="pmt1_norm", y="base_norm", color="type", symbol="type", size="Size",
                        hover_data=["key", "Years", "cost", "pmt1", "discount", "eff_cost"],
                        title="1. מפת יעילות (נמוך יותר = טוב יותר)",
                        labels={"pmt1_norm": "החזר ראשון (מנורמל)", "base_norm": "עלות כוללת (מנורמלת)"},
                        color_discrete_sequence=px.colors.qualitative.Bold  # Brighter colors
                    )
                    # Add chosen markers
                    sel_df = df_debug[df_debug["selected"]==True]
                    if not sel_df.empty:
                        fig1.add_trace(go.Scatter(
                            x=sel_df["pmt1_norm"], y=sel_df["base_norm"],
                            mode='markers', marker=dict(size=20, color="red", symbol="circle-open", line=dict(width=3)),
                            name="נבחר", showlegend=False
                        ))
                    st.plotly_chart(fig1, width="stretch")

                # Chart 2
                with d2:
                    fig2 = px.scatter(
                        df_debug, x="Years", y="eff_cost", color="type", symbol="type", size="Size",
                        hover_data=["key", "base_norm", "pmt1_norm", "discount"],
                        title="2. ציון משוקלל לפי אורך הלוואה",
                         labels={"Years": "שנים", "eff_cost": "ציון (עלות אפקטיבית)"},
                         color_discrete_sequence=px.colors.qualitative.Bold # Brighter colors
                    )
                    # Add chosen markers
                    if not sel_df.empty:
                        fig2.add_trace(go.Scatter(
                            x=sel_df["Years"], y=sel_df["eff_cost"],
                            mode='markers', marker=dict(size=20, color="red", symbol="circle-open", line=dict(width=3)),
                            name="נבחר", showlegend=False
                        ))
                    st.plotly_chart(fig2, width="stretch")

    if st.button("⬅️ החלף את התמהיל הידני בתוצאה האופטימלית", key="apply_optimal_tab2"):
        if opt is None:
             st.warning("לא נמצאה תוצאה אופטימלית להחלה.")
        else:
             sol_tracks = opt["tracks"]
             st.session_state.mix_tracks = [
                {
                    "uid": f"opt{i+1}",
                    "principal": float(tr["principal"]),
                    "months": int(tr["months"]),
                    "rate": float(tr["rate"]),
                    "rate_type": tr["rate_type"],
                    "freq": int(tr["freq"]),
                    "schedule_t": tr["schedule_t"],
                }
                for i, tr in enumerate(sol_tracks)
             ]
             st.success("התמהיל הידני הוחלף בתוצאה האופטימלית.")
             st.rerun()

with tab3:
    st.subheader("מחזור משכנתא")

    # --- 1. הגדרת פונקציות עזר (חייבות להופיע בראש הטאב) ---
    def plot_table_and_graf(data):
            df_routes_var = pd.DataFrame([val[0] for val in data])
            st.data_editor(
                df_routes_var,
                hide_index=True,
                width="stretch",
                key=f"data_editor_tab3_{hash(str(df_routes_var))}",  # מפתח ייחודי
                column_config={
                    "מסלול": st.column_config.SelectboxColumn(
                        "מסלול", options=df_routes_var['מסלול'].unique().tolist()
                    ),
                    "סכום": st.column_config.NumberColumn("סכום", step=1000, format="₪%d"),
                    "תקופה (חודשים)": st.column_config.NumberColumn("תקופה (חודשים)", step=1),
                    "ריבית": st.column_config.NumberColumn("ריבית (%)", format="%.2f"),
                    "החזר ראשון": st.column_config.NumberColumn("החזר ראשון", step=1),
                }
            )

            # חישובים מצטברים
            current_mortage = []
            max_len = 0
            for i, sch in enumerate([val[1] for val in data]):
                max_len = max(max_len, len(sch))
                current_mortage.append((i, sch))

            L = max_len
            X = list(range(1, L+1))
            total_payment = [0.0] * L
            total_interest = [0.0] * L
            total_principal_base = [0.0] * L
            total_indexation = [0.0] * L
            total_used_rate = [0.0] * L

            for _, sch in current_mortage:
                xs, opening, payment, interest, principal_base, indexation, closing, used_rate = _schedule_arrays(sch)
                total_payment = [a+b for a, b in zip(total_payment, _pad(payment, L))]
                total_interest = [a+b for a, b in zip(total_interest, _pad(interest, L))]
                total_principal_base = [a+b for a, b in zip(total_principal_base, _pad(principal_base, L))]
                total_indexation = [a+b for a, b in zip(total_indexation, _pad(indexation, L))]
                total_used_rate = [a+b for a, b in zip(total_used_rate, _pad(used_rate, L))]

            for m in range(L):
                total_used_rate[m] = total_used_rate[m] / len(current_mortage)

            # סיכומים
            total_P, total_I, total_K = 0, 0, 0
            for _, sch in current_mortage:
                P, I, K = summarize_schedule(sch)
                total_P += P
                total_I += I
                total_K += K

            col1, col2, col3, col4, col5 = st.columns(5)
            #col1.metric("סכום קרן שנותר", f"₪{total_P:,.0f}")
            col1.metric("יתרה לסילוק כולל עמלות", f"₪{total_P:,.0f}")
            col2.metric("סה״כ ריבית", f"₪{total_I:,.0f}")
            col3.metric("סה״כ הצמדה", f"₪{total_K:,.0f}")
            col4.metric("סך התשלומים", f"₪{(total_P+total_I+total_K):,.0f}")
            col5.metric("החזר ראשון", f"₪{total_payment[0]:,.0f}")

            # גרף החזר חודשי
            fig_tot_pay = go.Figure()
            fig_tot_pay.add_trace(go.Scatter(x=X, y=total_payment, mode="lines", name="תשלום חודשי כולל"))
            fig_tot_pay.update_layout(title="החזר חודשי כולל", xaxis_title="חודש", yaxis_title="₪")

            # גרף חלוקת תשלום שנתי
            years = (L-1)//12 + 1
            y_idx = list(range(1, years+1))
            pr_y, in_y, idx_y = [0.0]*years, [0.0]*years, [0.0]*years
            for m in range(1, L+1):
                y = (m-1)//12
                pr_y[y] += total_principal_base[m-1]
                in_y[y] += total_interest[m-1]
                idx_y[y] += total_indexation[m-1]

            fig_tot_br = go.Figure()
            fig_tot_br.add_trace(go.Bar(x=y_idx, y=pr_y, name="קרן", marker=dict(color='blue')))
            fig_tot_br.add_trace(go.Bar(x=y_idx, y=idx_y, name="הצמדה", marker=dict(color='green')))
            fig_tot_br.add_trace(go.Bar(x=y_idx, y=in_y, name="ריבית", marker=dict(color='red')))
            fig_tot_br.update_layout(title="חלוקת התשלום השנתי", barmode="stack", xaxis_title="שנה", yaxis_title="₪")

            # גרף ריבית אפקטיבית
            fig_used_rate = go.Figure()
            fig_used_rate.add_trace(go.Scatter(x=X, y=total_used_rate, mode="lines", name="הריבית האפקטיבית"))
            fig_used_rate.update_layout(title="הריבית האפקטיבית", xaxis_title="חודש", yaxis_title="%")

            c1, c2, c3 = st.columns(3)
            with c1: st.plotly_chart(fig_tot_pay, width="stretch")
            with c2: st.plotly_chart(fig_tot_br, width="stretch")
            with c3: st.plotly_chart(fig_used_rate, width="stretch")

    def summarize_mortgage(data, label, base_total=None):
            """מחזיר סיכום מספרי של תרחיש"""
            current_mortage = [(i, sch) for i, sch in enumerate([val[1] for val in data])]
            L = max(len(sch) for _, sch in current_mortage)

            total_payment = [0.0] * L
            #max_payment_per_track = []
            for _, sch in current_mortage:
                _, _, payment, _, _, _, _, _ = _schedule_arrays(sch)
                total_payment = [a+b for a, b in zip(total_payment, _pad(payment, L))]
                #max_payment_per_track.append(max(payment))
            max_payment = max(total_payment)
            total_P, total_I, total_K = 0, 0, 0
            for _, sch in current_mortage:
                P, I, K = summarize_schedule(sch)
                total_P += P
                total_I += I
                total_K += K

            total_sum = total_P + total_I + total_K
            first_payment = total_payment[0]

            # חיסכון מול התרחיש הבסיסי
            saving_abs = 0
            saving_pct = 0
            if base_total:
                saving_abs = base_total - total_sum
                saving_pct = (saving_abs / base_total) * 100

            return {
                "תרחיש": label,
                "החזר חודשי ראשון": first_payment,
                "החזר חודשי מקסימלי": max_payment,
                "סך הכל תשלומים": total_sum,
                "חיסכון ₪": saving_abs,
                "חיסכון %": saving_pct
            }, total_payment



    # אתחול ה-State אם הוא לא קיים
    if "cache_id" not in st.session_state:
        st.session_state.cache_id = 0
    if "last_file_hash" not in st.session_state:
        st.session_state.last_file_hash = None

    col_header, col_reset = st.columns([4, 1])
    with col_header:
        st.subheader("מחזור משכנתא")


    uploaded_file = st.file_uploader("העלה קובץ json", type=["json"], key="pdf_up_tab3")
    capital_allocation = st.selectbox("הקצאת הון (LTV)", options=["35%", "50%", "60%", "75%", "100%", "any"], key="cap_tab3")

    if uploaded_file and capital_allocation:
        file_bytes = uploaded_file.getvalue()

        # הרצת החישוב עם ה-cache_id הנוכחי
        with st.spinner("מבצע חישובים"):
            print("file_bytes")
            api_json = json.loads(file_bytes.decode("utf-8-sig"))
            print("api_json")
            tracks = convert_api_json_to_loan_tracks(api_json)
            print("tracks")
            ori_m, upd_m, non_m, opt_m = create_4_candidate_mortages(tracks, capital_allocation)
            print("ori_m, upd_m, non_m, opt_m ")
            best_res = find_best_mortage(tracks, capital_allocation)

        
        options_map = {
            "משכנתא נוכחית": ori_m, "משכנתא מעודכנת": upd_m,
            "משכנתא לא צמודה": non_m, "משכנתא מחזור אופטימלי": opt_m
        }

        # --- 3. באנר ROBIN (מנצח) ---
        total_pay_ori = sum(t[0]['סך כל התשלומים'] for t in ori_m)
        if best_res:
            best_mort, total_pay_best, first_pay_best, max_pay_best, best_name = best_res
            st.markdown(f"""<div style="background-color:#2E4C2E; color:white; padding:20px; border-radius:15px; text-align:center;">
                <h1 style='color:#76ff03;'>חיסכון של ₪{total_pay_ori - total_pay_best:,.0f}</h1>
                <p>התמהיל המנצח: {best_name}</p></div>""", unsafe_allow_html=True)

        # --- 4. השוואה (Multiselect) ---
        # --- 4. השוואה (Multiselect) ---
        selected = list(options_map.keys())

        if selected:
            fig_comp = go.Figure()
            summaries = []
            loan_total = sum(t[0]['סכום'] for t in ori_m)
            
            # שמירת זרם התשלומים של המקור להשוואה
            _, series_ori = summarize_mortgage(ori_m, "משכנתא נוכחית", total_pay_ori)
            
            # פונקציית עזר לסיכום שנתי
            def to_yearly(monthly_list):
                return [sum(monthly_list[i:i+12]) for i in range(0, len(monthly_list), 12)]
            
            yearly_ori = to_yearly(series_ori)
            all_yearly_savings = {} # מילון לשמירת החיסכון השנתי של כל בחירה

            for opt in selected:
                summary, series = summarize_mortgage(options_map[opt], opt, total_pay_ori)
                summary['החזר לשקל'] = summary["סך הכל תשלומים"] / loan_total
                summaries.append(summary)
                
                # הוספת קו לגרף ההחזרים החודשיים
                fig_comp.add_trace(go.Scatter(y=series, mode="lines", name=f"החזר חודשי: {opt}"))
                
                # חישוב חיסכון שנתי מול המקור
                y_series = to_yearly(series)
                max_y = max(len(yearly_ori), len(y_series))
                
                # השלמת אפסים להשוואה
                temp_ori = yearly_ori + [0] * (max_y - len(yearly_ori))
                temp_opt = y_series + [0] * (max_y - len(y_series))
                
                # חישוב חיסכון שנתי (מקור פחות הצעה)
                all_yearly_savings[opt] = [orig - sug for orig, sug in zip(temp_ori, temp_opt)]
            
            st.write("### 📉 השוואת החזרים חודשיים")
            st.plotly_chart(fig_comp, use_container_width=True)

            # --- 5. הטבלה המבוקשת ---
            st.subheader("טבלה השוואתית")
            st.dataframe(pd.DataFrame(summaries).style.format({
                "החזר חודשי ראשון": "₪{:,.0f}", "החזר חודשי מקסימלי": "₪{:,.0f}",
                "סך הכל תשלומים": "₪{:,.0f}", "חיסכון ₪": "₪{:,.0f}",
                "חיסכון %": "{:.1f}%", "החזר לשקל": "{:.2f}"
            }), use_container_width=True)

            # --- 6. גרף חיסכון שנתי להשוואה (התוספת החדשה) ---
            st.divider()
            st.subheader("💰 חיסכון שנתי לכל הצעה (מול המצב הקיים)")
            
            fig_savings = go.Figure()
            
            for label, savings in all_yearly_savings.items():
                if label == "משכנתא נוכחית": continue # אין טעם להשוות את המקור לעצמו
                
                years_axis = list(range(1, len(savings) + 1))
                fig_savings.add_trace(go.Bar(
                    x=years_axis, 
                    y=savings, 
                    name=f"חיסכון ב-{label}"
                ))

            fig_savings.update_layout(
                title="כמה כסף נשאר לכם בכיס בכל שנה?",
                xaxis_title="שנה",
                yaxis_title="₪ חיסכון שנתי",
                barmode='group', # מציג עמודות אחת ליד השנייה
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_savings, use_container_width=True)
            
            # בונוס: טבלת חיסכון מצטבר
            with st.expander("לצפייה בחיסכון המצטבר לפי שנים"):
                cum_data = {"שנה": list(range(1, max(len(s) for s in all_yearly_savings.values()) + 1))}
                for label, savings in all_yearly_savings.items():
                    if label == "משכנתא נוכחית": continue
                    current_sum = 0
                    cum_list = []
                    for s in savings:
                        current_sum += s
                        cum_list.append(current_sum)
                    cum_data[label] = cum_list
                
                st.dataframe(pd.DataFrame(cum_data).style.format("₪{:,.0f}"), use_container_width=True)

        # --- 6. פירוט מלא (Expander) ---
        st.divider()
        for label, data in options_map.items():
            with st.expander(f"פירוט: {label}"):
                plot_table_and_graf(data)
         
with tab4:
    # ----- גרף אינפלציה -----
    infl_monthly = DATASTORE.infl_monthly
    infl_series_yearly = [monthly_to_yearly(float(i)) * 100 for i in infl_monthly]
    fig_infl = go.Figure()
    fig_infl.add_trace(go.Scatter(x=list(range(1, len(infl_series_yearly) + 1)), y=infl_series_yearly, mode="lines", name="אינפלציה"))
    fig_infl.update_layout(title="אינפלציה חודשית (מומר לשנתי)", xaxis_title="חודש", yaxis_title="שיעור (%)")
    st.plotly_chart(fig_infl, width="stretch", key="inflation_tab4")

    # ----- איור עוגן – NOMINAL -----
    zero_series_nom = DATASTORE.zero_nominal.tolist()
    zero_pct   = [100 * v for v in zero_series_nom]
    base_zero  = zero_pct[0]
    zero_pp    = [v - base_zero for v in zero_pct]
    V_VALUES = [12, 18, 24, 30, 36, 48, 60, 84, 120]
    fig_anchor_multi_NOMINAL = go.Figure()
    for V in V_VALUES:
        anchor_path = build_anchor(zero_series_nom, V=V, term_m=con.HORIZON)
        base_val = anchor_path[0] * 100.0
        anchor_pp = [(a * 100.0 - base_val) for a in anchor_path]
        fig_anchor_multi_NOMINAL.add_trace(
            go.Scatter(x=list(range(1, con.HORIZON + 1)), y=anchor_pp, mode="lines", name=f"V={V}", line=dict(shape="hv"))
        )
    fig_anchor_multi_NOMINAL.add_trace(go.Scatter(x=list(range(1, con.HORIZON + 1)), y=zero_pp, mode="lines", name="Zero nominal (Δpp)"))
    fig_anchor_multi_NOMINAL.update_layout(title="איור עוגן נומינלי – השוואה בין מרווחי חידוש (Δ נק׳ אחוז)", xaxis_title="חודש", yaxis_title="שינוי (נק׳ אחוז)")
    st.plotly_chart(fig_anchor_multi_NOMINAL, width="stretch", key="anchor_nominal_tab4")

    # ----- איור עוגן – REAL -----
    zero_series_real = DATASTORE.zero_real.tolist()
    zero_pct   = [100 * v for v in zero_series_real]
    base_zero  = zero_pct[0]
    zero_pp    = [v - base_zero for v in zero_pct]
    fig_anchor_multi_REAL = go.Figure()
    for V in V_VALUES:
        anchor_path = build_anchor(zero_series_real, V=V, term_m=con.HORIZON)
        base_val = anchor_path[0] * 100.0
        anchor_pp = [(a * 100.0 - base_val) for a in anchor_path]
        fig_anchor_multi_REAL.add_trace(
            go.Scatter(x=list(range(1, con.HORIZON + 1)), y=anchor_pp, mode="lines", name=f"V={V}", line=dict(shape="hv"))
        )
    fig_anchor_multi_REAL.add_trace(go.Scatter(x=list(range(1, con.HORIZON + 1)), y=zero_pp, mode="lines", name="Zero real (Δpp)"))
    fig_anchor_multi_REAL.update_layout(title="איור עוגן ריאלי – השוואה בין מרווחי חידוש (Δ נק׳ אחוז)", xaxis_title="חודש", yaxis_title="שינוי (נק׳ אחוז)")
    st.plotly_chart(fig_anchor_multi_REAL, width="stretch", key="anchor_real_tab4")

with tab5:
    st.header("⚙️ הגדרות מערכת מתקדמות")
    st.markdown("שינוי פרמטרים גלובליים וטבלאות. **זהירות**: שינוויים כאן נשמרים לדיסק ומשפיעים קבוע.")
    
    with st.form("config_form_advanced"):
        
        # --- Scalars ---
        st.subheader("1. פרמטרים בסיסיים")
        c1, c2, c3, c4 = st.columns(4)
        new_boi = c1.number_input("ריבית בנק ישראל (%)", value=float(con.bank_of_israel_rate), step=0.05, format="%.2f")
        new_margin = c2.number_input("מרווח פריים (%)", value=float(con.prime_margin), step=0.05, format="%.2f")
        new_max_share = c3.number_input("Max Share (0-1)", value=float(con.MAX_SHARE_PER_OPTION), step=0.05)
        new_min_share = c4.number_input("Min Active Share", value=float(con.MIN_ACTIVE_SHARE), step=0.05)
        
        c5, c6 = st.columns(2)
        new_internal_ratio = c5.number_input("יחס לימיט פנימי (החזר/הכנסה)", value=float(con.INTERNAL_ratio_limit), step=0.01)
        
        st.subheader('ריביות תמהיל אופטימלי - תוספת אחוזים יחסית')
        new_manual_increase = st.number_input("תוספת ריבית יחסית (%)", value=float(getattr(con, 'manual_interest_increase', 0.0)), step=0.5, help="אחוז להוספה לריבית שנמצאה באופטימיזציה (למשל 10% יוסיף 0.5 לריבית של 5)")

        capital_opts = ["35%", "50%", "60%", "75%", "100%", "any"]
        try:
            curr_alloc = str(con.defult_capital_allocation)
            def_idx = capital_opts.index(curr_alloc)
        except ValueError:
            def_idx = 0
        new_default_ltv = c6.selectbox("LTV ברירת מחדל", capital_opts, index=def_idx)

        st.divider()
        c7, c8 = st.columns(2)
        with c7:
            new_no_saving_flag = c7.number_input("מינימום חיסכון", value=float(con.no_savings), step=0.1)
        with c8:
            new_diff_between_opt =  c8.number_input("הפרש להעדפה של חסכון פנימי וחיצוני", value=float(con.diff_between_opt), step=0.1) 
        st.divider()

        # --- Tables ---
        st.subheader("2. טבלאות ריבית (קל\"צ / ק\"צ)")
        
        def edit_rate_table(table_dict, label):
            st.caption(label)
            # Convert to list of dicts for data_editor
            data = [{"Years": k, "Rate": v} for k, v in table_dict.items()]
            df = pd.DataFrame(data)
            edited_df = st.data_editor(df, key=label, num_rows="dynamic", width="stretch")
            # Convert back
            new_dict = {}
            for _, row in edited_df.iterrows():
                try:
                    y = int(row["Years"])
                    r = float(row["Rate"])
                    new_dict[y] = r
                except:
                    pass
            return dict(sorted(new_dict.items()))

        c_tbl1, c_tbl2 = st.columns(2)
        with c_tbl1:
            new_fixed_non_indexed = edit_rate_table(con.FIXED_NON_INDEXED_TABLE, "קל\"צ (FIXED NON-INDEXED)")
        with c_tbl2:
            new_fixed_indexed = edit_rate_table(con.FIXED_INDEXED_TABLE, "ק\"צ (FIXED INDEXED)")

        st.divider()

        # --- Spreads ---
        st.subheader("3. מרווחים (Spreads)")
        spread_data = [{"Track": k, "Spread": v} for k, v in con.SPREADS.items()]
        spread_df = pd.DataFrame(spread_data)
        edited_spread_df = st.data_editor(spread_df, key="edit_spreads", num_rows="dynamic", width="stretch")
        new_spreads = {row["Track"]: float(row["Spread"]) for _, row in edited_spread_df.iterrows()}

        st.divider()
        
        # --- Loan Adj Rules ---
        st.subheader("4. התאמות LTV (Loan Adj Rules)")
        all_ltvs = set()
        for t_adj in con.LOAN_ADJ_RULES.values():
            all_ltvs.update(t_adj.keys())
        sorted_ltvs = sorted(list(all_ltvs)) 
        
        adj_rows = []
        for track_name, rules in con.LOAN_ADJ_RULES.items():
            row = {"Track": track_name}
            for ltv in sorted_ltvs:
                row[ltv] = rules.get(ltv, 0.0)
            adj_rows.append(row)
            
        adj_df = pd.DataFrame(adj_rows)
        # Reorder columns
        cols = ["Track"] + [c for c in adj_df.columns if c != "Track"]
        adj_df = adj_df[cols]
        
        edited_adj_df = st.data_editor(adj_df, key="edit_adj_rules", num_rows="dynamic", width="stretch")
        
        new_loan_adj_rules = {}
        for _, row in edited_adj_df.iterrows():
            t_name = row["Track"]
            if not t_name: continue
            new_rules = {}
            for col in edited_adj_df.columns:
                if col != "Track":
                    try:
                        new_rules[col] = float(row[col])
                    except:
                        new_rules[col] = 0.0
            new_loan_adj_rules[t_name] = new_rules

        # --- Discount Params ---
        import importlib
        importlib.reload(con)
        st.divider()
        st.subheader("5. פרמטרים לאופטימיזציה (Discount Params)")
        
        # Sensitivity
        st.markdown("**הנחת רגישות (Sensitivity Discount)**")
        # Defaults if missing to avoid crash on first render
        def_sens = getattr(con, "SENSITIVITY_DISCOUNT", {"נמוך":0, "בינוני":25, "גבוה":40})
        sens_data = [{"Level": k, "Discount": v} for k, v in def_sens.items()]
        sens_df = pd.DataFrame(sens_data)
        # Use new keys to force refresh of columns
        edited_sens_df = st.data_editor(sens_df, key="edit_sens_disc_v2", num_rows="dynamic", width="stretch")
        new_sens_disc = {row["Level"]: float(row["Discount"]) for _, row in edited_sens_df.iterrows()}

        st.markdown("**הנחת פירעון מוקדם (Prepay Discount)**")
        # Structure is {Scenario: { 'משתנה': {1:X, ..., 60:Y}, 'קבועה': Z }}
        def_prepay = getattr(con, "PREPAY_DISCOUNT", {})
        
        # 1. Collect all unique frequencies from the config
        all_freqs = set()
        for p_vals in def_prepay.values():
            var_part = p_vals.get("משתנה", {})
            if isinstance(var_part, dict):
                all_freqs.update(var_part.keys())
        
        # Sort freqs numerically
        sorted_freqs = sorted(list(all_freqs)) if all_freqs else [1, 12, 24, 30, 36, 60, 84, 120]

        # 2. Build rows
        prepay_rows = []
        for p_key, p_vals in def_prepay.items():
            row = {"Scenario": p_key}
            
            var_dict = p_vals.get("משתנה", {})
            # If flat float (legacy), convert to dict
            if not isinstance(var_dict, dict):
                val = float(var_dict)
                var_dict = {f: val for f in sorted_freqs}

            # Add columns for each freq
            for f in sorted_freqs:
                row[f"Var_{f}m"] = float(var_dict.get(f, 0))
            
            row["Fixed_Discount"] = float(p_vals.get("קבועה", 0))
            prepay_rows.append(row)
        
        prepay_df = pd.DataFrame(prepay_rows)
        
        # Reorder columns: Scenario, then Var columns sorted, then Fixed
        col_order = ["Scenario"] + [f"Var_{f}m" for f in sorted_freqs] + ["Fixed_Discount"]
        # Ensure only existing columns are selected (in case dataframe creation added others?? unlikely)
        prepay_df = prepay_df[col_order]

        edited_prepay_df = st.data_editor(prepay_df, key="edit_prepay_disc_v3", num_rows="dynamic", width="stretch")
        
        # 3. Reconstruct
        new_prepay_disc = {}
        for _, row in edited_prepay_df.iterrows():
            scen = row["Scenario"]
            fixed_d = float(row["Fixed_Discount"])
            
            # Rebuild variable dict
            var_map = {}
            for f in sorted_freqs:
                col_name = f"Var_{f}m"
                if col_name in row:
                    var_map[f] = float(row[col_name])
            
            new_prepay_disc[scen] = {
                "משתנה": var_map,
                "קבועה": fixed_d
            }

        # --- Submit ---
        st.markdown("---")
        submitted = st.form_submit_button("💾 שמור את כל ההגדרות (כולל טבלאות)")
        
        if submitted:
            updates = {
                "bank_of_israel_rate": new_boi,
                "prime_margin": new_margin,
                "MAX_SHARE_PER_OPTION": new_max_share,
                "MIN_ACTIVE_SHARE": new_min_share,
                "INTERNAL_ratio_limit": new_internal_ratio,
                "defult_capital_allocation": new_default_ltv,
                "manual_interest_increase": new_manual_increase,
                "FIXED_NON_INDEXED_TABLE": new_fixed_non_indexed,
                "FIXED_INDEXED_TABLE": new_fixed_indexed,
                "SPREADS": new_spreads,
                "LOAN_ADJ_RULES": new_loan_adj_rules,
                "SENSITIVITY_DISCOUNT": new_sens_disc,
                "PREPAY_DISCOUNT": new_prepay_disc,
                "no_savings":new_no_saving_flag,
                "diff_between_opt":new_diff_between_opt
            }
            save_config_to_file(updates)
            sync_ok, sync_error = sync_config_to_api(updates)
            
            # Explicitly reload the config module to ensure new values are picked up
            # Robust reload logic:
            #if con.__name__ in sys.modules:
            #    importlib.reload(con)
            #else:
                # If module is missing from sys.modules (e.g. Windows/Streamlit quirk), re-import it
             #   import config_v2
                # We can't easily rebind the global 'con' here without declaring global, 
                # but since we are about to rerun, the file update is what matters.
                # However, to be safe and satisfy "Must reload":
              #  pass

            if sync_ok:
                st.success("ההגדרות נשמרו בהצלחה וסונכרנו ל-API! מבצע רענון...")
            else:
                st.warning(f"ההגדרות נשמרו מקומית, אך סנכרון API נכשל: {sync_error}")
            import time
            time.sleep(1)
            st.rerun()

with tab6:
    uploaded_file_tab6 = st.file_uploader("העלה קובץ json", type=["json"], key="pdf_up_tab6")
    
    @st.cache_data
    def parse_api_json(file_content):
        try:
            return json.loads(file_content.decode("utf-8-sig"))
        except Exception as e:
            st.error(f"Cannot parse JSON file: {e}")
            return None

    if uploaded_file_tab6 is not None:
        api_json_first_loan = parse_api_json(uploaded_file_tab6.getvalue())
        first_loan_tracks = convert_api_json_to_first_loan_tracks(api_json_first_loan)
        
        all_scenarios = {}
        original_principal = 0
        monthly_payment_orig = 0
        max_period_orig = 0

        # 1. עיבוד מסלולים מקוריים
        original_tracks_results = []
        ori_display_data = []
        for i, tr in enumerate(first_loan_tracks):
            freq_val = tr.get("freq")
            prime_offset = 0
            if tr["rate_type"] in ("מלצ","מצ",'מטח דולר','מטח יורו'):
                freq_val = int(freq_val) if freq_val else 60
            elif tr["rate_type"] == "פריים":
                freq_val = 1
                prime_offset = tr["rate"] - (con.bank_of_israel_rate + con.prime_margin)
            
            sch = calculate_schedule(tr["principal"], tr["months"], tr["rate"], tr["schedule_t"], tr["rate_type"], freq_val, con.prime_margin, prime_offset)
            original_tracks_results.append((f"orig_{i}", sch))
            original_principal += tr["principal"]
            monthly_payment_orig += sch[0][2]
            max_period_orig = max(max_period_orig, tr["months"])
            ori_display_data.append({
                        "סוג מסלול": tr["rate_type"],
                        "סכום (₪)": f"{tr['principal']:,.0f}",
                        "תקופה (חודשים)": tr["months"],
                        "ריבית (%)": f"{tr['rate']:.2f}%",
                        "החזר חודשי": f"{sch[0][2]:,.0f} ₪"
                    })
        
        all_scenarios["תמהיל מוצע (מהקובץ)"] = original_tracks_results
        
        # 2. הרצת אופטימיזציה
        with st.spinner("מחשב תמהיל אופטימלי להשוואה..."):
            opt_sol, opt_totals, opt_err = optimize_mortgage(
                float(original_principal),#loan_amount=
                con.defult_capital_allocation,#ltv_input='75%',
                float(monthly_payment_orig * con.monthly_income_factor),#monthly_income_net=monthly_payment_orig * 3,
                con.sensitivity,#sensitivity=
                con.prepay_window_key,
                con.durations_months(max_period_orig),#durations_months=[m for m in range(12, max_period_orig, 1)],
                con.bank_of_israel_rate,
                con.prime_margin,
                con.objective_mode,
                con.alpha,
                monthly_payment_orig,
                None # routes_data_ori=
            )
            
            if not opt_err:
                optimal_tracks_results = []
                opt_display_data = []
                for i, tr in enumerate(opt_sol):
                    optimal_tracks_results.append((f"opt_{i}", tr["schedule"]))
                    opt_display_data.append({
                        "סוג מסלול": tr["rate_type"],
                        "סכום (₪)": f"{tr['principal']:,.0f}",
                        "תקופה (חודשים)": tr["months"],
                        "ריבית (%)": f"{tr['rate']:.2f}%",
                        "החזר חודשי": f"{tr['schedule'][0][2]:,.0f} ₪"
                    })
                all_scenarios["הסל האופטימלי"] = optimal_tracks_results

        # 3. לולאת תצוגה והשוואה
        scenario_summaries = {}
        for name, tracks in all_scenarios.items():
            st.markdown(f"### {name}")
            
            # חישוב נתונים מסכמים לסנריו
            t_p, t_i, t_k = 0, 0, 0
            first_pmt_sum = 0
            max_months = 0
            
            for _, sch in tracks:
                p, i, k = summarize_schedule(sch)
                t_p += p; t_i += i; t_k += k
                first_pmt_sum += sch[0][2]
                max_months = max(max_months, len(sch))
            
            # חישוב החזר מקסימלי לאורך כל חיי המשכנתא
            max_pmt = 0
            for m_idx in range(max_months):
                current_month_total = sum(sch[m_idx][2] for _, sch in tracks if m_idx < len(sch))
                if current_month_total > max_pmt:
                    max_pmt = current_month_total
            
            total_cost = t_p + t_i + t_k
            scenario_summaries[name] = {"total": total_cost}

            # תצוגת מטריקות
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("סכום הלוואה", f"₪{original_principal:,.0f}")
            m_col2.metric("תקופה מקסימלית", f"{max_months // 12} שנים ({max_months} חודשים)")
            m_col3.metric("סה-כ החזר כולל", f"₪{total_cost:,.0f}")

            m_col4, m_col5 = st.columns(2)
            m_col4.metric("החזר חודשי ראשון", f"₪{first_pmt_sum:,.0f}")
            m_col5.metric("החזר חודשי מקסימלי", f"₪{max_pmt:,.0f}", delta=f"{max_pmt - first_pmt_sum:,.0f}+", delta_color="inverse")

            # הצגת פירוט למסלול אופטימלי
            if name == "הסל האופטימלי" and not opt_err:
                st.write("**פירוט מסלולי התמהיל האופטימלי:**")
                st.table(opt_display_data)
            if name == "תמהיל מוצע (מהקובץ)" and not opt_err:
                st.write("**פירוט מסלולי התמהיל המוצע:**")
                st.table(ori_display_data)
            st.divider()

        

        # 4. סיכום חיסכון
        if len(scenario_summaries) > 1:
            best_name = min(scenario_summaries, key=lambda x: scenario_summaries[x]["total"])
            worst_name = max(scenario_summaries, key=lambda x: scenario_summaries[x]["total"])
            
            money_save = scenario_summaries[worst_name]['total'] - scenario_summaries[best_name]['total']
            
            st.balloons()
            st.success(f"🏆 **התמהיל המשתלם ביותר:** {best_name}")
            st.metric("פוטנציאל חיסכון כולל", f"{money_save:,.0f} ₪", delta_color="normal")           

with tab7:
    st.markdown("<h3 style='text-align: center; color: #E67E22;'>מחשבון משכנתא - סלים אחידים</h3>", unsafe_allow_html=True)
    
    # כניסת נתונים
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        sim_p = st.select_slider("סכום ההלוואה (₪)", options=list(range(150000, 4000001, 50000)), value=1000000)
    with col_input2:
        sim_y = st.select_slider("תקופת ההחזר (שנים)", options=list(range(4, 31)), value=20)
    
    sim_m = sim_y * 12

    # --- חישוב מהיר (שולף מהקאש אם אין שינוי) ---
    all_schedules = get_all_schedules(sim_p, sim_m)

    # בחירת הסל להצגה ויזואלית
    selected_sal = st.radio("בחר סל להצגה:", list(all_schedules.keys()), horizontal=True)
    current_tracks = all_schedules[selected_sal]

    # --- מכאן והלאה קוד התצוגה שלך (ללא שינוי לוגי) ---
    total_pay, total_int, total_idx, first_pmt = 0, 0, 0, 0
    for track in current_tracks:
        p, i, k = summarize_schedule(track["sch"])
        total_pay += (p + i + k)
        total_int += i
        total_idx += k
        first_pmt += track["sch"][0][2]
    avg_rate = sum(t["rate"] * (t["principal"]/sim_p) for t in current_tracks)

    # --- 5. תצוגת כרטיסיות נתונים ---
    st.markdown("""
        <style>
        .result-box { background-color: #E8F6F3; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-right: 6px solid #27AE60; font-size: 16px; }
        .label { float: right; font-weight: bold; color: #34495E; }
        .value { float: left; font-weight: bold; color: #27AE60; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='result-box'><span class='label'>סכום ההלוואה</span><span class='value'>₪{sim_p:,.0f}</span><div style='clear:both;'></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'><span class='label'>סה\"כ החזר משוער</span><span class='value'>₪{total_pay:,.0f}</span><div style='clear:both;'></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'><span class='label'>מזה הצמדה למדד</span><span class='value'>₪{total_idx:,.0f}</span><div style='clear:both;'></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'><span class='label'>החזר חודשי ראשון</span><span class='value'>₪{first_pmt:,.0f}</span><div style='clear:both;'></div></div>", unsafe_allow_html=True)

    # --- 6. פירוט מסלולים ---
    st.write("### פירוט מסלולים")
    for t in current_tracks:
        st.markdown(f"""
            <div style="background-color: #E67E22; color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold; width: 35%; text-align: right;">{t['name']}</div>
                <div style="text-align: center; width: 25%;">ריבית: {t['rate']:.2f}%</div>
                <div style="text-align: left; width: 40%;">סכום: ₪{t['principal']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- 7. גרף זרם תשלומים משולש (קרן, ריבית, הצמדה) ---
    st.write("### התפתחות החזרים לאורך זמן")
    months_axis = list(range(1, sim_m + 1))
    
    # חישוב זרמים מצטברים
    principal_flow = [sum(t["sch"][m-1][5] for t in current_tracks if m <= len(t["sch"])) for m in months_axis]
    interest_flow = [sum(t["sch"][m-1][3] for t in current_tracks if m <= len(t["sch"])) for m in months_axis]
    indexation_flow = [sum(t["sch"][m-1][6] for t in current_tracks if m <= len(t["sch"])) for m in months_axis]

    fig_robin = go.Figure()
    fig_robin.add_trace(go.Scatter(x=months_axis, y=principal_flow, mode='lines', name='פירעון קרן', line=dict(color="#2CB164", width=3)))
    fig_robin.add_trace(go.Scatter(x=months_axis, y=interest_flow, mode='lines', name='תשלום ריבית', line=dict(color='#E67E22', width=3)))
    fig_robin.add_trace(go.Scatter(x=months_axis, y=indexation_flow, mode='lines', name='רכיב הצמדה', line=dict(color='#3498DB', width=3, dash='dot')))
    
    fig_robin.update_layout(
        xaxis_title="חודשים", yaxis_title="₪",
        hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_robin, use_container_width=True)
