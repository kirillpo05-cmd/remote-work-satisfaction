# Decision log

Format: context, options, decision, rationale, cost, status.
An entry without a cost is incomplete.

---

## D-000: Specification before code
Context: a scoped assignment evaluated on structure and reasoning rather than
volume of code.
Options: (a) grow the product out of a notebook; (b) write the module
specification first and build against it.
Decision: (b).
Rationale: the brief grades engineering quality and deployability explicitly;
notebook-first work historically produces a monolith with no module boundaries.
Cost: the first working hours produce no visible output, and the spec has to be
updated whenever an interface changes.
Status: accepted

---

## D-001: Product framing follows the signal check
Context: <fill in with the actual Phase 0 verdict>
Options: (a) a driver-explanation product; (b) a verification product that
protects the user from a false conclusion.
Decision: <fill in>
Rationale: <fill in>
Cost: <fill in>
Status: accepted

---

<!-- Decisions that must be recorded as the build proceeds:

- Ordinal versus multinomial target treatment
- Including or excluding Work_Location == Onsite from a question about REMOTE
  work satisfaction
- Treating nulls in Mental_Health_Condition as "no condition" rather than missing
- Permutation importance instead of feature_importances_
- Streamlit instead of a JS frontend
- Splitting the API from the UI
- Committing the dataset instead of requiring Kaggle credentials
- The "simplest model that ties" selection rule
-->
