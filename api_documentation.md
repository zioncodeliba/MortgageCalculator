# Mortgage API Server Documentation

This document describes the API endpoints for the Mortgage API Server.

## Base URL
The API runs on `http://localhost:8000` (default for FastAPI/Uvicorn).

## Endpoints

### 1. `POST /simulate`

Simulates mortgage tracks based on user input.

**Request Body:**
- `data`: List of dictionaries, where each dictionary represents a mortgage track.
    - `principal` (float): The loan amount for the track.
    - `months` (int): Duration of the loan in months.
    - `rate` (float): Annual interest rate (percentage).
    - `rate_type` (string): Type of interest rate (e.g., 'קלצ', 'פריים', 'משתנה צמודה').
    - `freq` (int, optional): Frequency of interest rate changes in months (for variable rates).
    - `schedule_t` (string): Type of amortization schedule (e.g., 'שפיצר', 'קרן שווה').
- `property_value` (float, query parameter): Value of the property.

**Response:**
Returns a JSON object containing:
```json
{
  "individual_tracks": [
    {
      "id": 1,
      "summary": {
        "principal_paid": 500000.0,
        "interest_paid": 120000.0,
        "indexation_paid": 0.0,
        "total_repayment": 620000.0,
        "first_payment": 2500.0
      },
      "graph_data": {
        "months": [1, 2, 3, ...],
        "opening_balance": [500000.0, 498000.0, ...],
        "monthly_payment": [2500.0, 2500.0, ...],
        "interest_component": [1000.0, 990.0, ...],
        "closing_balance": [498000.0, 496000.0, ...],
        "effective_rate": [4.0, 4.0, ...],
        "yearly_distribution": {
            "principal": [10000.0, ...],
            "interest": [5000.0, ...],
            "indexation": [0.0, ...]
        }
      },
      "full_schedule": [
        [1, 500000.0, 2500.0, 1000.0, 1500.0, 1500.0, 0.0, 498500.0, 4.0],
        ...
      ]
    }
  ],
  "combined": {
    "summary": {
      "total_principal": 500000.0,
      "total_interest": 120000.0,
      "total_indexation": 0.0,
      "total_repayment": 620000.0,
      "max_payment": 2500.0,
      "first_payment": 2500.0
    },
    "graph_data": {
      "months": [1, 2, 3, ...],
      "monthly_payment": [2500.0, ...],
      "opening_balance": [500000.0, ...],
      "closing_balance": [498500.0, ...],
      "interest_balance": [1000.0, ...]
    }
  }
}
```

---

### 2. `POST /optimize`

Finds the optimal mortgage mix based on user constraints.

**Request Body (Form Data):**
- `loan` (float): Total loan amount requested.
- `property_value` (float): Value of the property.
- `income` (float): Net monthly income.
- `max_pmt` (float): Maximum allowed monthly payment.
- `max_scan_years` (float): Maximum loan duration in years to consider.

**Response:**
Returns a JSON object containing:
```json
{
  "tracks": [
    {
      "principal": 300000.0,
      "months": 360,
      "rate": 6.0,
      "rate_type": "פריים",
      "freq": 1,
      "schedule_t": "שפיצר",
      "schedule": [
        [1, 300000.0, 1800.0, 1500.0, 300.0, 300.0, 0.0, 299700.0, 6.0],
        ...
      ] 
    },
    ...
  ],
  "max_payment": 5500.0
}
```
**Schedule Columns:**
The `schedule` array contains rows with the following columns (by index):
0. Month (int)
1. Opening Balance (float)
2. Monthly Payment (float)
3. Interest Payment (float)
4. Principal Payment (Indexed) (float)
5. Principal Payment (Base) (float)
6. Indexation Payment (float)
7. Closing Balance (float)
8. Effective Yearly Rate (float)

```json
{
  "error": "Error description..."
}
```

---

### 3. `POST /refinance`

Analyzes refinancing options by comparing existing mortgage files (JSON) with current market rates.

**Request Body:**
- `json_file` (UploadFile): A JSON file containing existing mortgage details.
- `property_value` (float, form data): Current value of the property.

**Response:**
Returns a JSON object containing:
```json
{
  "comparison_table": [
    {
      "תרחיש": "משכנתא נוכחית",
      "החזר חודשי ראשון": 5000.0,
      "החזר חודשי מקסימלי": 5500.0,
      "סך הכל תשלומים": 1200000.0,
      "חיסכון ₪": 0,
      "החזר לשקל": 1.2,
      "internal_rate_of_return": 3.5,
      "הפרש החזר חודשי ראשון": 0,
      "חיסכון חודשי ממוצע": 0
    },
    {
      "תרחיש": "משכנתא אופטימלית",
      "החזר חודשי ראשון": 4500.0,
      ...
    }
  ],
  "detailed_scenarios": {
    "משכנתא נוכחית": {
      "summary": { ... },
      "tracks": [ ... ],
      "combined_graph": {
        "months": [ ... ],
        "payments": [ ... ]
      }
    },
    ...
  },
  "ltv_used": "60%"
}
```

---

### 4. `POST /approval-check`

Checks approval feasibility and compares a proposed mix (from JSON) against an optimal mix.

**Request Body:**
- `json_file` (UploadFile): A JSON file containing the proposed mortgage mix.
- `property_value` (float, form data): Value of the property.

**Response:**
Returns a JSON object containing:
```json
{
  "proposed_mix": {
    "metrics": {
      "סכום_הלוואה": 1000000.0,
      "תקופה_מקסימלית": "20 שנים (240 חודשים)",
      "סהכ_החזר_כולל": 1500000.0,
      "החזר_חודשי_ראשון": 5000.0,
      "החזר_חודשי_מקסימלי": 5500.0,
      "delta_pmt": 500.0
    },
    "label": "תמהיל מוצע (מהקובץ)"
  },
  "optimal_mix": {
    "metrics": { ... },
    "table": [
      {
        "סוג_מסלול": "פריים",
        "סכום": 330000.0,
        "תקופה_חודשים": 360,
        "ריבית": 6.0,
        "החזר_חודשי": 1800.0
      },
      ...
    ],
    "label": "הסל האופטימלי",
    "error": null
  },
  "savings": {
    "total_savings": 50000.0,
    "best_name": "הסל האופטימלי"
  }
}
```

---

### 5. `GET /uniform-baskets`

Analyzes three "uniform" mortgage baskets (standard mixes) for a given loan amount and duration.

**Query Parameters:**
- `principal` (float): Total loan amount.
- `years` (int): Loan duration in years.

**Response:**
Returns a JSON object with keys `סל אחיד 1`, `סל אחיד 2`, `סל אחיד 3`. Each basket contains:
```json
{
  "summary": {
    "סכום_הלוואה": 1000000.0,
    "סהכ_החזר_משוער": 1600000.0,
    "מזה_הצמדה_למדד": 200000.0,
    "החזר_חודשי_ראשון": 5500.0
  },
  "tracks_detail": [
    {
      "שם": "קלצ מלא",
      "ריבית": 4.5,
      "סכום": 1000000.0
    }
  ],
  "graph_data": {
    "months": [1, 2, ...],
    "principal_repayment": [ ... ],
    "interest_payment": [ ... ],
    "indexation_component": [ ... ]
  }
}
```


---

