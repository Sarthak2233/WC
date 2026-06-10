# Skill: Diagnostic Loop
Name: diagnostic-loop
Description: A meta-skill used to debug harness failures and improve agent reliability by attributing errors to specific harness subsystems.

## Instructions
When a task fails or an agent produces an incorrect result, you must activate the Diagnostic Loop:
1.  **Observe the Failure:** Identify the exact point of failure (e.g., wrong file edited, logic error in feature synthesis).
2.  **Attribute to Subsystem:** Map the failure to one of the 5 Harness Subsystems:
    *   **Instructions:** Was the guidance ambiguous or missing?
    *   **State:** Was the agent operating on stale context or incorrect milestone data?
    *   **Verification:** Did a sensor fail to catch the error?
    *   **Scope:** Did the agent drift into unrelated files or logic?
    *   **Environment:** Was there a missing dependency or tool failure?
3.  **Fix the Harness:** Do not just fix the code. **Update the relevant .antigravity file** (AGENT.md, CONVENTIONS.md, etc.) to prevent the failure from recurring.
4.  **Re-Execute:** Run the task again within the improved harness.

## Triggers
- Activate automatically after any test failure or user-reported bug.
- Triggered by `/diagnostic-loop`.
