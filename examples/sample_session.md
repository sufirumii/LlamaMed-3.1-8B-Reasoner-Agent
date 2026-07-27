# Sample session

```
$ llamamed-agent ingest data/pdfs/discharge_summary.pdf
  discharge_summary.pdf: 14 chunks
Done. Index saved to data/index

$ llamamed-agent chat
you> This patient's creatinine is 1.4 mg/dL, they're a 62-year-old woman.
     What's their estimated GFR, and does the discharge summary mention
     any existing kidney diagnosis?

Thought: I should compute the eGFR first, then check the indexed document
         for any kidney-related diagnosis.
Action: clinical_calculator
Action Input: {"calculation": "egfr_ckd_epi_2021", "args": {"creatinine_mg_dl": 1.4, "age_years": 62, "sex": "female"}}

Observation: 41.2 mL/min/1.73m^2 (formula: 2021 CKD-EPI creatinine equation (race-free))

Thought: That's consistent with CKD stage 3. Let me check the discharge
         summary for a documented diagnosis.
Action: search_documents
Action Input: {"query": "chronic kidney disease diagnosis"}

Observation: [discharge_summary.pdf, p.2, score=0.81]
Assessment: 62F with history of hypertension, presenting with fatigue.
Labs notable for elevated creatinine. Chronic kidney disease, stage 3,
suspected -- nephrology referral placed.

Thought: I have enough to answer.
Final Answer: Estimated GFR is about 41 mL/min/1.73m^2 (2021 CKD-EPI
creatinine equation), consistent with CKD stage 3 (moderate reduction).
The discharge summary (p.2) does note a suspected CKD stage 3 diagnosis
with a nephrology referral already placed, which lines up with this
calculation. This is a research tool output, not a clinical determination
-- please confirm with the treating clinician.
```
