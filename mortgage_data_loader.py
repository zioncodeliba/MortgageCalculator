# mortgage_data_loader.py
import os
import re
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import streamlit as st
from dataclasses import dataclass
from typing import List, Tuple
import config as con
import typing as t
from pathlib import Path
import json
from datetime import datetime

# ================================================================
#  נתוני עקום תשואות, מק"ם, וקובצי אקסל — הורדה מהאינטרנט
# ================================================================
@st.cache_resource(show_spinner=True, ttl=60 * 60 * 12)
def load_boi_data():
    """
    טוען את כל הנתונים מבנק ישראל (API + Excel) ושומר במטמון.
    במקרה של כשל, טוען את הגיבוי האחרון הקיים בתיקייה.
    """
    base_dir = Path(__file__).parent.resolve()
    backup_dir = base_dir / "last_boi_anchors"
    backup_dir.mkdir(parents=True, exist_ok=True)  # וידוא שהתיקייה קיימת
    
    # נתיב לקובץ של היום ספציפית
    today_str = datetime.now().strftime("%d-%m-%Y")
    backup_path = backup_dir / f"last_boi_anchors_{today_str}.json"
    
    try:
        print("🔄 Fetching data from Bank of Israel...")

        # === עקום תשואות נומינלי וריאלי ===
        url_yields = (
            "https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/"
            "BOI.STATISTICS/ZCM/1.0/?c%5BSERIES_CODE%5D="
            "ZC_TSB_ZND_01Y_MA,ZC_TSB_ZND_02Y_MA,ZC_TSB_ZND_03Y_MA,"
            "ZC_TSB_ZND_04Y_MA,ZC_TSB_ZND_05Y_MA,ZC_TSB_ZND_07Y_MA,"
            "ZC_TSB_ZND_10Y_MA,ZC_TSB_ZND_15Y_MA,"
            "ZC_TSB_ZRD_01Y_MA,ZC_TSB_ZRD_02Y_MA,ZC_TSB_ZRD_03Y_MA,"
            "ZC_TSB_ZRD_04Y_MA,ZC_TSB_ZRD_05Y_MA,ZC_TSB_ZRD_07Y_MA,"
            "ZC_TSB_ZRD_10Y_MA,ZC_TSB_ZRD_15Y_MA,ZC_TSB_ZRD_20Y_MA"
            "&lastNObservations=1&locale=en"
        )
        resp = requests.get(url_yields, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {
            "msg": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
            "ss": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/structurespecific",
        }

        nominal_anchor, real_anchor = {}, {}
        for series in root.findall(".//Series", ns):
            code = series.attrib.get("SERIES_CODE")
            obs = series.find("Obs", ns)
            if obs is not None:
                value = obs.attrib.get("OBS_VALUE")
                if not value: continue
                # המרה למפתחות אינטגרליים (שנים)
                key = int(code.split("_")[3][:2]) # חילוץ מספר השנים
                if "ZRD" in code:
                    real_anchor[key] = float(value)
                elif "ZND" in code:
                    nominal_anchor[key] = float(value)

        # === נתוני מק"ם ===
        url_makam = (
            "https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/"
            "BOI.STATISTICS/SECDWH/1.0/"
            "DWH_SRC_0351.D.YTM.GB_MK.BS114.NI._Z.M012.S121._Z._Z.W2.B08.C.AVG_W_MV?locale=he"
        )
        resp = requests.get(url_makam, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        data = [
            (pd.to_datetime(obs.attrib.get("TIME_PERIOD")), float(obs.attrib.get("OBS_VALUE")))
            for obs in root.iter("Obs") if obs.attrib.get("OBS_VALUE")
        ]
        df = pd.DataFrame(data, columns=["תאריך", "ערך"]).sort_values("תאריך")
        makam_anchor = float(df["ערך"].iloc[-1])

        boi_data = {
            "nominal_anchor": nominal_anchor,
            "real_anchor": real_anchor,
            "makam_anchor": makam_anchor,
            "last_updated": today_str
        }
        
        # שמירה לגיבוי הצליחה
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(boi_data, f, ensure_ascii=False, indent=4)
            
        return boi_data

    except Exception as e:
        st.warning(f"⚠️ שרת בנק ישראל לא זמין. מחפש נתונים שמורים...")
        
        # חיפוש כל קבצי ה-JSON בתיקייה
        backup_files = list(backup_dir.glob("last_boi_anchors_*.json"))
        
        if backup_files:
            # מיון לפי זמן שינוי הקובץ (החדש ביותר ראשון)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_backup = backup_files[0]
            
            with open(latest_backup, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.info(f"✅ נטענו נתונים מתאריך: {data.get('last_updated', 'לא ידוע')}")
                return data
        else:
            st.error("❌ לא נמצא קובץ גיבוי כלשהו במערכת.")
            return None
        
@st.cache_resource(show_spinner=True, ttl=60 * 60 * 12)
def fetch_latest_boi_excels():
    """
    מאתר ומוריד את קובצי האקסל. אם נכשל, מחזיר את הקבצים האחרונים מהתיקייה המקומית.
    """
    base_dir = Path(__file__).parent.resolve()
    download_dir = base_dir / "boi_yields"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("🔍 Searching for Excel files on BOI site...")
        url = "https://www.boi.org.il/roles/statistics/makamandbonds/yield/#mainContent"
        
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a", href=True)
        excel_links = {}

        for a in links:
            text = a.get_text(" ", strip=True)
            href = a["href"]
            if href.endswith((".xls", ".xlsx")):
                date_match = re.search(r"(\d{2}[./]\d{2}[./]\d{4})", text)
                date_str = date_match.group(1).replace(".", "-").replace("/", "-") if date_match else "no_date"
                
                file_type = href.split('.')[-1]
                if "התשואה הנומינלית" in text:
                    excel_links["nominal"] = (href, date_str, file_type)
                elif "התשואה הריאלית" in text:
                    excel_links["real"] = (href, date_str, file_type)
                elif "נתוני התשואות הנומינליות והריאליות" in text:
                    excel_links["model"] = (href, date_str, file_type)

        if not excel_links:
            raise RuntimeError("לא נמצאו קישורים תקינים באתר.")

        saved_paths = {}
        for name, (href, date_str, file_type) in excel_links.items():
            file_url = href if href.startswith("http") else "https://www.boi.org.il" + href
            if date_str == 'no_date':
                date_str = "manual_" + datetime.now().strftime("%d-%m-%Y")
            
            file_path = download_dir / f"{name}_{date_str}.{file_type}"
            
            # הורדה בפועל
            print(f"⬇️ Downloading {name}...")
            r = requests.get(file_url, timeout=20)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)
            saved_paths[name] = str(file_path)

        return saved_paths["model"], saved_paths["nominal"], saved_paths["real"]

    except Exception as e:
        st.warning(f"⚠️ שגיאה בהורדת קבצים מבנק ישראל: {e}")
        st.info("🔄 מנסה לטעון קבצים קיימים מהמאגר המקומי...")
        
        fallback_paths = {}
        for category in ["model", "nominal", "real"]:
            # מחפש את כל הקבצים שמתחילים בשם הקטגוריה (למשל nominal_*.xlsx)
            files = list(download_dir.glob(f"{category}_*.*"))
            if files:
                # מיון לפי זמן שינוי אחרון - הכי חדש ראשון
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                fallback_paths[category] = str(files[0])
            else:
                fallback_paths[category] = None

        # בדיקה אם חסר אחד מהקבצים הקריטיים
        if not all(fallback_paths.values()):
            missing = [k for k, v in fallback_paths.items() if v is None]
            st.error(f"❌ לא נמצאו קבצי גיבוי עבור: {', '.join(missing)}")
            raise RuntimeError("No local backup files available.")

        st.success(f"✅ נטענו קבצי גיבוי מקומיים.")
        return fallback_paths["model"], fallback_paths["nominal"], fallback_paths["real"]
# ================================================================
#  קריאת נתונים מאקסל — WorkbookLoader & DataStore
# ================================================================
def _locate_header_row(df_raw: pd.DataFrame, search_rows: int = 15) -> int:
    """מזהה שורת כותרת (מכילה 'שנה' ו'חודש'); אם לא נמצא — fallback ל-4."""
    for i in range(min(search_rows, len(df_raw))):
        row = df_raw.iloc[i].astype(str).fillna("").tolist()
        if ("שנה" in row) and ("חודש" in row):
            return i
    return 4

def _numeric_maturity_columns(df: pd.DataFrame) -> List[str]:
    """עמודות מח"מ מספריות (1..360)."""
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except Exception:
            pass
    return sorted(cols, key=lambda x: float(x))

def _to_decimal_if_needed(arr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(arr.astype(float))
    if np.isfinite(med) and med > 1.0:
        return arr / 100.0
    return arr

@dataclass
class DataStore:
    zero_nominal: np.ndarray        # אורך HORIZON, תשואת זירו שנתית בעשרוני
    zero_real: np.ndarray           # אורך HORIZON, תשואת זירו שנתית בעשרוני
    infl_monthly: List[float]       # אינפלציה חודשית בעשרוני (ΔK/K)
    infl_K: List[float]             # K_t מצטבר (יחסי לבסיס)
    boi_forward_annual_pct: List[float]  # ריבית 1M-forward שנתית באחוזים (360 ערכים)

class WorkbookLoader:
    def __init__(self, file_path: str, scenario: str = con.SCENARIO):
        self.file_path = file_path
        self.scenario = scenario

    def _read_sheet(self, sheet_name: str, header=None) -> pd.DataFrame:
        return pd.read_excel(self.file_path, sheet_name=sheet_name, header=header)

    def _latest_row_by_date(self, df_raw: pd.DataFrame) -> pd.Series:
        """מקבל DataFrame ללא כותרת, מאתר את כותרת העמודות, מסנן לפי SCENARIO אם קיים,
        ובוחר את השורה האחרונה לפי תאריך (שנה+חודש) אם ניתן, אחרת אחרונה."""
        header_row = _locate_header_row(df_raw)
        headers = df_raw.iloc[header_row].tolist()
        df = df_raw.iloc[header_row + 1:].copy()
        df.columns = headers
        df = df.dropna(how="all", axis=1)

        # סינון לפי תרחיש אם העמודה קיימת
        # סינון לפי תרחיש – ניקח רק "מדדי" או "קלנדרי"
        if con.SCENARIO_COL in df.columns:
            df = df[df[con.SCENARIO_COL].isin(["מדדי", "קלנדרי"])].copy()


        # ניסיון יצירת עמודת תאריך
        if "שנה" in df.columns and "חודש" in df.columns:
            try:
                df["שנה"] = pd.to_numeric(df["שנה"], errors="coerce").astype("Int64")
            except Exception:
                pass
            if df["חודש"].dtype == object:
                df["חודש_num"] = df["חודש"].map(con.HE_MONTHS).fillna(pd.to_numeric(df["חודש"], errors="coerce"))
            else:
                df["חודש_num"] = df["חודש"]
            df = df[df["שנה"].notna() & df["חודש_num"].notna()].copy()
            if not df.empty:
                df["תאריך"] = pd.to_datetime(
                    dict(year=df["שנה"].astype(int), month=df["חודש_num"].astype(int), day=1),
                    errors='coerce'
                )
                df = df[df["תאריך"].notna()].copy()

        latest = df.sort_values("תאריך").iloc[-1] if ("תאריך" in df.columns and not df.empty) else df.iloc[-1]
        #print(latest)
        return latest

    def _extract_zero_series(self, latest_row: pd.Series, horizon: int) -> np.ndarray:
        """מחלץ סדרת זירו (בד"כ עמודות '1'..'360'), ממיר לעשרוני, משלים/גוזר לאורך horizon."""
        df = latest_row.to_frame().T
        mat_cols = _numeric_maturity_columns(df)
        if not mat_cols:
            raise ValueError("לא נמצאו עמודות מח\"מ מספריות בגיליון.")
        vals = pd.to_numeric(df.iloc[0][mat_cols], errors="coerce").to_numpy(dtype=float)
        vals = _to_decimal_if_needed(vals)
        series = vals.tolist()
        if len(series) >= horizon:
            series = series[:horizon]
        else:
            series += [series[-1]] * (horizon - len(series))
        return np.array(series, dtype=float)

    def _compute_forward_from_zero(self, z: np.ndarray) -> List[float]:
        """מקבל z(T) שנתי בעשרוני (T בחודשים, 1..360) ומחזיר 1M-forward שנתית באחוזים (360 ערכים)."""
        wanted_months = np.arange(1, len(z) + 1, dtype=int)
        fwd = np.empty_like(z)
        prev_T = 0.0
        prev_z = np.nan
        for i, m in enumerate(wanted_months):
            T = m / 12.0
            zT = z[i]
            if m == 1:
                fwd[i] = zT
            else:
                dlt = T - prev_T
                fwd[i] = ((1.0 + zT) ** T / (1.0 + prev_z) ** prev_T) ** (1.0 / dlt) - 1.0
            prev_T, prev_z = T, zT
        return (fwd * 100.0).astype(float).tolist()

    def _compute_inflation_from_gap(self) -> t.Tuple[List[float], List[float]]:
        df = self._read_sheet(con.SHEET_GAP, header=0)
        if con.SCENARIO_COL in df.columns and self.scenario is not None:
            df = df[df[con.SCENARIO_COL] == self.scenario]
        else:
            if con.SCENARIO_COL in df.columns:
                df = df[df[con.SCENARIO_COL].isin(["מדדי", "קלנדרי"])]
        if df.empty:
            raise ValueError("לא נמצאו נתונים מתאימים בגליון הפערים")
        last_row = df.iloc[-1]
        cpi_series = last_row.loc['Unnamed: 3':'Unnamed: 363'].reset_index(drop=True)
        cpi_values = cpi_series.tolist()
        Periodic_inflation, Change_compared_to_the_base_index = [], []
        for i in range(1, len(cpi_values)):
            prev, curr = cpi_values[i - 1], cpi_values[i]
            if pd.notna(prev) and pd.notna(curr) and prev != 0:
                Periodic_inflation.append((float(curr) / float(prev)) - 1.0)
                Change_compared_to_the_base_index.append(float(curr) / float(cpi_values[0]))
        return Periodic_inflation, Change_compared_to_the_base_index

    def load(self, horizon: int) -> DataStore:
        df_nom_raw = self._read_sheet(con.SHEET_NOMINAL, header=None)
        latest_nom = self._latest_row_by_date(df_nom_raw)
        zero_nominal = self._extract_zero_series(latest_nom, horizon)

        df_real_raw = self._read_sheet(con.SHEET_REAL, header=None)
        latest_real = self._latest_row_by_date(df_real_raw)
        zero_real = self._extract_zero_series(latest_real, horizon)

        infl_monthly, infl_K = self._compute_inflation_from_gap()
        boi_forward_annual_pct = self._compute_forward_from_zero(zero_nominal)
        #import pdb; pdb.set_trace()
        return DataStore(
            zero_nominal=zero_nominal,
            zero_real=zero_real,
            infl_monthly=infl_monthly,
            infl_K=infl_K,
            boi_forward_annual_pct=boi_forward_annual_pct,
        )

@st.cache_resource(show_spinner=False)
def load_workbook_data(file_path: str, horizon: int, scenario: str) -> DataStore:
    """קריאה יחידה לאקסל + עיבוד, נשמרת בזיכרון של האפליקציה.
    המפתח של המטמון כולל את מועד העדכון האחרון של הקובץ (mtime)."""
    loader = WorkbookLoader(file_path, scenario=scenario)
    return loader.load(horizon)






