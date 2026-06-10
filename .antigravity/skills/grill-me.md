# Skill: Grill Me
Name: grill-me
Description: Proactively identifies edge cases and resolves ambiguities before planning or coding.

## Instructions
When this skill is activated, you must:
1. **Analyze the Request:** Identify what is missing or underspecified.
2. **Interview the User:** Ask 3-5 sharp, technical questions to resolve:
    - **Edge Cases:** What happens if data for a country is missing for a specific year?
    - **Constraints:** What are the performance limits for this specific ETL step?
    - **Assumptions:** What is the specific heuristic for the "Legacy Burden" in this context?
3. **Summarize:** After the user answers, provide a "Final Intent" summary before proceeding to planning or TDD.

## Triggers
- Use before starting any new Milestone or complex sub-phase.
- Triggered by `/grill-me`.
