"""Synthetic Indian T2D / prediabetes cohort: demographics, genetics, medications and the
hidden physiology each virtual patient runs on. Also exports HL7 FHIR R4 bundles (ABDM-ready
resource shapes) so the EHR stream looks like what a hospital system would actually send.

Prior distributions are loosely anchored on ICMR-INDIAB (2023) and published South Asian
T2D phenotype studies (lower BMI at onset, higher insulin resistance, TCF7L2 risk allele
frequency ~0.28). All people here are fictional.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .physiology import TruePhysiology

FIRST_M = ["Arjun", "Rahul", "Vikram", "Suresh", "Karthik", "Imran", "Rohit", "Manoj", "Prakash", "Anil",
           "Deepak", "Sanjay", "Harish", "Naveen", "Faisal", "Gurpreet", "Rajesh", "Venkat", "Abhishek", "Joseph"]
FIRST_F = ["Priya", "Lakshmi", "Anjali", "Sunita", "Kavya", "Fatima", "Meena", "Divya", "Pooja", "Revathi",
           "Neha", "Shalini", "Asha", "Nandini", "Geeta", "Harpreet", "Sneha", "Mary", "Bhavana", "Rekha"]
SURNAMES_N = ["Sharma", "Verma", "Gupta", "Singh", "Yadav", "Mishra", "Khan", "Agarwal", "Chauhan", "Malhotra"]
SURNAMES_S = ["Iyer", "Reddy", "Nair", "Rao", "Menon", "Pillai", "Gowda", "Krishnan", "Hegde", "Shetty"]
CITIES_N = ["Delhi", "Lucknow", "Jaipur", "Chandigarh", "Patna", "Kolkata"]
CITIES_S = ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Mysuru", "Coimbatore"]


@dataclass
class Lifestyle:
    region: str
    wake_mean_min: float
    sleep_mean_h: float
    sleep_sd_h: float
    bad_night_p: float
    daily_steps: float
    morning_walk_p: float
    post_meal_walk_p: float
    snack_p: float
    sweet_p: float
    skip_breakfast_p: float
    log_compliance: float
    stress_day_p: float
    rhr: float
    hrv_base: float
    meal_prefs: dict = field(default_factory=dict)


@dataclass
class PatientRecord:
    patient_id: str
    abha_id: str
    name: str
    sex: str
    age: int
    city: str
    region: str
    height_cm: float
    weight_kg: float
    bmi: float
    status: str                     # "T2D" | "Prediabetes"
    diabetes_duration_y: float
    family_history: bool
    tcf7l2_rs7903146: int           # risk allele count 0/1/2
    prs_z: float                    # polygenic risk score (z-score)
    hypertension: bool
    dyslipidemia: bool
    sbp: int
    dbp: int
    ldl: int
    egfr: int
    metformin: bool
    sulfonylurea: bool
    basal_insulin: bool
    gb_untreated: float
    hba1c: float = float("nan")     # filled after simulation (ADAG from mean glucose)
    fpg: float = float("nan")
    hba1c_history: list = field(default_factory=list)

    def to_flat(self) -> dict:
        d = asdict(self)
        d.pop("hba1c_history")
        return d


def _abha(rng: np.random.Generator) -> str:
    digits = "".join(str(x) for x in rng.integers(0, 10, 12))
    return f"91-{digits[:4]}-{digits[4:8]}-{digits[8:]}"


def sample_patient(i: int, rng: np.random.Generator) -> tuple[PatientRecord, TruePhysiology, Lifestyle]:
    status = "T2D" if rng.random() < 0.75 else "Prediabetes"
    sex = "M" if rng.random() < 0.5 else "F"
    region = "south" if rng.random() < 0.55 else "north"
    age = int(np.clip(rng.normal(53 if status == "T2D" else 43, 10), 28, 78))
    height = float(np.clip(rng.normal(168 if sex == "M" else 155, 6.5), 145, 190))
    bmi = float(np.clip(rng.normal(26.3, 3.8), 19, 40))
    weight = bmi * (height / 100) ** 2
    duration = float(np.clip(rng.gamma(2.0, 3.5), 0.2, 25)) if status == "T2D" else 0.0
    tcf = int(rng.binomial(2, 0.28))
    prs = float(rng.normal(0.3 if status == "T2D" else 0.0, 1.0))
    fam = bool(rng.random() < (0.6 if status == "T2D" else 0.45))

    # Untreated physiology
    if status == "T2D":
        gb0 = float(np.clip(122 + 4.0 * duration ** 0.75 + 4 * prs + 0.8 * (bmi - 25) + rng.normal(0, 9), 105, 230))
        kx = 1.25e-4 * np.exp(-0.035 * duration) * (1 - 0.09 * tcf) * np.exp(-0.10 * prs) \
            * np.exp(-0.025 * (bmi - 25)) * rng.lognormal(0, 0.22)
        sg = float(np.clip(rng.normal(0.012, 0.0025), 0.006, 0.022))
        dawn = float(rng.uniform(6, 30))
    else:
        gb0 = float(np.clip(rng.normal(106, 6), 95, 125))
        kx = 3.4e-4 * (1 - 0.06 * tcf) * np.exp(-0.08 * prs) * np.exp(-0.02 * (bmi - 25)) * rng.lognormal(0, 0.2)
        sg = float(np.clip(rng.normal(0.016, 0.003), 0.008, 0.026))
        dawn = float(rng.uniform(0, 10))

    # Medication assignment, then treatment effects
    metformin = status == "T2D" and rng.random() < 0.9
    su = status == "T2D" and gb0 > 145 and rng.random() < 0.55
    insulin = status == "T2D" and gb0 > 170 and rng.random() < 0.6
    gb = gb0 * (0.86 if metformin else 1.0) * (0.92 if insulin else 1.0)
    kx *= 1.12 if metformin else 1.0
    drug_x = (0.0022 if su else 0.0) + (0.0016 if insulin else 0.0)

    phys = TruePhysiology(
        weight_kg=weight, gb=gb, sg=sg, kx=float(kx),
        p2=float(0.028 * rng.lognormal(0, 0.18)),
        gut_scale=float(rng.lognormal(0, 0.13)),
        k_ex=float(9e-5 * rng.lognormal(0, 0.3)),
        dawn_amp=dawn,
        sleep_sens=float(rng.uniform(0.12, 0.45)),
        stress_gb=float(rng.uniform(6, 22)),
        drug_x=drug_x,
    )

    htn = bool(rng.random() < np.clip(0.2 + 0.012 * (age - 35) + 0.02 * (bmi - 25), 0.05, 0.8))
    dys = bool(rng.random() < 0.45)
    sbp = int(rng.normal(142 if htn else 124, 9))
    dbp = int(rng.normal(90 if htn else 79, 6))
    ldl = int(np.clip(rng.normal(128 if dys else 104, 25), 50, 220))
    egfr = int(np.clip(108 - 0.75 * (age - 30) - 0.9 * duration + rng.normal(0, 9), 28, 125))

    if region == "south":
        name = f"{rng.choice(FIRST_M if sex == 'M' else FIRST_F)} {rng.choice(SURNAMES_S)}"
        city = str(rng.choice(CITIES_S))
    else:
        name = f"{rng.choice(FIRST_M if sex == 'M' else FIRST_F)} {rng.choice(SURNAMES_N)}"
        city = str(rng.choice(CITIES_N))

    rec = PatientRecord(
        patient_id=f"P{i:03d}", abha_id=_abha(rng), name=name, sex=sex, age=age, city=city, region=region,
        height_cm=round(height, 1), weight_kg=round(weight, 1), bmi=round(bmi, 1), status=status,
        diabetes_duration_y=round(duration, 1), family_history=fam, tcf7l2_rs7903146=tcf, prs_z=round(prs, 2),
        hypertension=htn, dyslipidemia=dys, sbp=sbp, dbp=dbp, ldl=ldl, egfr=egfr,
        metformin=bool(metformin), sulfonylurea=bool(su), basal_insulin=bool(insulin), gb_untreated=round(gb0, 1),
    )

    sleep_mean = float(np.clip(rng.normal(6.6, 0.7), 5.0, 8.2))
    life = Lifestyle(
        region=region,
        wake_mean_min=float(rng.uniform(5.75, 7.75) * 60),
        sleep_mean_h=sleep_mean,
        sleep_sd_h=float(rng.uniform(0.4, 0.9)),
        bad_night_p=float(rng.uniform(0.05, 0.25)),
        daily_steps=float(np.clip(rng.normal(5500, 2200), 1800, 12000)),
        morning_walk_p=float(rng.beta(1.5, 2.0)),
        post_meal_walk_p=float(rng.beta(1.2, 3.0)),
        snack_p=float(rng.uniform(0.4, 0.95)),
        sweet_p=float(rng.uniform(0.02, 0.2)),
        skip_breakfast_p=float(rng.uniform(0.0, 0.15)),
        log_compliance=float(rng.uniform(0.82, 0.98)),
        stress_day_p=float(rng.uniform(0.08, 0.3)),
        rhr=float(np.clip(rng.normal(74 if status == "T2D" else 70, 7), 55, 92)),
        hrv_base=float(np.clip(rng.normal(48 - 0.45 * (age - 30) - (6 if status == "T2D" else 0), 7), 12, 70)),
    )
    return rec, phys, life


def finalize_labs(rec: PatientRecord, mean_glucose: float, fasting_glucose: float, rng: np.random.Generator,
                  ref_date: date) -> None:
    """Derive labs consistent with the simulated glycaemia (ADAG: eAG = 28.7*A1c - 46.7)."""
    rec.hba1c = round(float((mean_glucose + 46.7) / 28.7 + rng.normal(0, 0.15)), 1)
    rec.fpg = round(float(fasting_glucose + rng.normal(0, 5)), 0)
    trend = rng.normal(0.25 if rec.status == "T2D" else 0.1, 0.3)   # %-points per year, pre-treatment drift
    hist = []
    for k in range(5, 0, -1):
        months = 3 * k
        val = rec.hba1c - trend * months / 12 + rng.normal(0, 0.15)
        hist.append({"date": (ref_date - timedelta(days=30 * months)).isoformat(), "hba1c": round(float(val), 1)})
    hist.append({"date": ref_date.isoformat(), "hba1c": rec.hba1c})
    rec.hba1c_history = hist


# --------------------------------------------------------------------------- FHIR R4 export

def _obs(pid: str, code: str, display: str, value: float, unit: str, when: str, category: str = "laboratory") -> dict:
    return {
        "resourceType": "Observation", "id": str(uuid.uuid4()), "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                  "code": category}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": when,
        "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
    }


def to_fhir_bundle(rec: PatientRecord, ref_date: date) -> dict:
    pid = rec.patient_id
    when = ref_date.isoformat()
    birth_year = ref_date.year - rec.age
    entries = [{
        "resourceType": "Patient", "id": pid,
        "meta": {"tag": [{"system": "urn:glucotwin", "code": "synthetic", "display": "Synthetic patient. Not a real person."}]},
        "identifier": [{"system": "https://healthid.ndhm.gov.in", "value": rec.abha_id, "type": {"text": "ABHA Number"}}],
        "name": [{"text": rec.name}], "gender": "male" if rec.sex == "M" else "female",
        "birthDate": f"{birth_year}-01-01", "address": [{"city": rec.city, "country": "IN"}],
    }]

    def cond(code, display, onset_years=None):
        c = {"resourceType": "Condition", "id": str(uuid.uuid4()),
             "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
             "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}]},
             "subject": {"reference": f"Patient/{pid}"}}
        if onset_years is not None:
            c["onsetDateTime"] = (ref_date - timedelta(days=int(365 * onset_years))).isoformat()
        return c

    if rec.status == "T2D":
        entries.append(cond("44054006", "Diabetes mellitus type 2", rec.diabetes_duration_y))
    else:
        entries.append(cond("714628002", "Prediabetes", 0.5))
    if rec.hypertension:
        entries.append(cond("38341003", "Hypertensive disorder"))
    if rec.dyslipidemia:
        entries.append(cond("370992007", "Dyslipidemia"))

    for h in rec.hba1c_history:
        entries.append(_obs(pid, "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", h["hba1c"], "%", h["date"]))
    entries += [
        _obs(pid, "1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", rec.fpg, "mg/dL", when),
        _obs(pid, "39156-5", "Body mass index (BMI)", rec.bmi, "kg/m2", when, "vital-signs"),
        _obs(pid, "29463-7", "Body weight", rec.weight_kg, "kg", when, "vital-signs"),
        _obs(pid, "8480-6", "Systolic blood pressure", rec.sbp, "mm[Hg]", when, "vital-signs"),
        _obs(pid, "8462-4", "Diastolic blood pressure", rec.dbp, "mm[Hg]", when, "vital-signs"),
        _obs(pid, "13457-7", "LDL cholesterol (calculated)", rec.ldl, "mg/dL", when),
        _obs(pid, "33914-3", "eGFR (MDRD)", rec.egfr, "mL/min/{1.73_m2}", when),
    ]
    entries.append({
        "resourceType": "Observation", "id": str(uuid.uuid4()), "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "69548-6", "display": "Genetic variant assessment"}]},
        "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": when,
        "valueCodeableConcept": {"text": f"TCF7L2 rs7903146 risk alleles: {rec.tcf7l2_rs7903146}"},
        "component": [{"code": {"text": "Type 2 diabetes polygenic risk score (z)"}, "valueQuantity": {"value": rec.prs_z}}],
    })

    def med(rxnorm, display, dose):
        return {"resourceType": "MedicationStatement", "id": str(uuid.uuid4()), "status": "active",
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                                          "code": rxnorm, "display": display}]},
                "subject": {"reference": f"Patient/{pid}"}, "dosage": [{"text": dose}]}

    if rec.metformin:
        entries.append(med("6809", "Metformin", "500 mg twice daily with meals"))
    if rec.sulfonylurea:
        entries.append(med("25789", "Glimepiride", "2 mg once daily before breakfast"))
    if rec.basal_insulin:
        entries.append(med("274783", "Insulin glargine", "14 units at bedtime"))

    return {"resourceType": "Bundle", "type": "collection", "id": f"bundle-{pid}",
            "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries]}


def write_fhir(rec: PatientRecord, ref_date: date, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{rec.patient_id}.json").write_text(json.dumps(to_fhir_bundle(rec, ref_date), indent=1))
