import time
import threading
import mortgage_data_loader as dl
import config as con

# ============================================================
# משתנים גלובליים (בדיוק כמו בקוד המקורי שלך)
# ============================================================

XLSX_PATH = None
DATASTORE = None
NOMINAL_ANCHOR = None
REAL_ANCHOR = None
MAKAM_ANCOR = None
LAST_UPDATE = None

XLSX_PATH = dl.fetch_latest_boi_excels()
print(f'exstract latest_boi_excels from: {XLSX_PATH}')
DATASTORE = dl.load_workbook_data(XLSX_PATH, con.HORIZON, con.SCENARIO)

ancors = dl.load_boi_data()
NOMINAL_ANCHOR = ancors["nominal_anchor"]
REAL_ANCHOR = ancors["real_anchor"]
MAKAM_ANCOR = ancors["makam_anchor"]
LAST_UPDATE = time.time()
# ============================================================
# פונקציית רענון — מעדכנת את כל קבצי הריבית והעוגנים
# ============================================================

def refresh_data():
    """
    מוריד את קבצי ה-BOI מהאינטרנט, טוען אותם למשתנים הגלובליים,
    ומעדכן את כל העוגנים והדאטהסטים.
    """
    global XLSX_PATH, DATASTORE, NOMINAL_ANCHOR, REAL_ANCHOR, MAKAM_ANCHOR, LAST_UPDATE

    print("⏳ Refreshing BOI data...")

    # 1. הורדת הקבצים החדשים מהבנק
    XLSX_PATH = dl.fetch_latest_boi_excels()

    # 2. טעינת ה-Workbook (דאטהסט ראשי)
    DATASTORE = dl.load_workbook_data(
        XLSX_PATH,
        con.HORIZON,
        con.SCENARIO
    )

    # 3. טעינת העוגנים (anchors)
    anchors = dl.load_boi_data()
    NOMINAL_ANCHOR = anchors["nominal_anchor"]
    REAL_ANCHOR = anchors["real_anchor"]
    MAKAM_ANCHOR = anchors["makam_anchor"]

    # 4. תיעוד זמן העדכון האחרון
    LAST_UPDATE = time.time()

    print(f"✅ BOI data refreshed successfully at {time.ctime(LAST_UPDATE)}")
    print(f"   Loaded from: {XLSX_PATH}")


# ============================================================
# תהליך רקע — מרענן כל X שניות (ברירת מחדל: שעה = 3600)
# ============================================================

def start_background_updater(interval_seconds=3600):
    """
    מפעיל thread שרץ ברקע ומרענן את הנתונים כל interval_seconds.
    """
    def loop():
        while True:
            try:
                refresh_data()
            except Exception as e:
                print(f"❌ Error while refreshing BOI data: {e}")
            time.sleep(interval_seconds)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"🔁 Background BOI updater started (every {interval_seconds} sec)")
