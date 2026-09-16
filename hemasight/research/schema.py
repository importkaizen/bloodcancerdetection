"""Versioned CBC features and units shared by preparation and benchmarking."""

SCHEMA_VERSION = "cbc_screening_v1"

# canonical name: (exact public CSV header, multiplier, canonical unit)
COLUMN_MAP = {
    "wbc": ("WBC(10^9/L)", 1.0, "10^9/L"),
    "rbc": ("RBC(10^12/L)", 1.0, "10^12/L"),
    "platelets": ("PLT(10^9/L)", 1.0, "10^9/L"),
    "hemoglobin": ("HGB(g/L)", 0.1, "g/dL"),
    "lymphocytes": ("LYMPH%(%)", 1.0, "%"),
    "hematocrit": ("HCT(%)", 1.0, "%"),
    "mcv": ("MCV(fL)", 1.0, "fL"),
    "mch": ("MCH(pg)", 1.0, "pg"),
    "mchc": ("MCHC(g/L)", 0.1, "g/dL"),
    "rdw_cv": ("RDW-CV(%)", 1.0, "%"),
    "neutrophils": ("NEUT#(10^9/L)", 1.0, "10^9/L"),
    "monocytes": ("MONO#(10^9/L)", 1.0, "10^9/L"),
    "eosinophils": ("EO#(10^9/L)", 1.0, "10^9/L"),
    "basophils": ("BASO#(10^9/L)", 1.0, "10^9/L"),
    "lymphocytes_absolute": ("LYMPH#(10^9/L)", 1.0, "10^9/L"),
}

FEATURE_SETS = {
    "basic_cbc": list(COLUMN_MAP)[:5],
    "expanded_cbc": list(COLUMN_MAP),
}
