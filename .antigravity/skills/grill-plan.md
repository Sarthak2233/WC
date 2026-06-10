# Skill: Grill Plan
Name: grill-plan
Description: Critiques an implementation plan for edge cases, DSA efficiency, and alignment with psychopolitical requirements.

## Instructions
When this skill is activated, you must:
1. **Analyze the Input Plan:** Read the proposed implementation details thoroughly.
2. **Audit for DSA Efficiency:** Identify potential $O(N^2)$ bottlenecks or memory-intensive operations. Suggest $O(N \log N)$ or $O(N)$ alternatives.
3. **Verify Psychopolitical Alignment:** Ensure the plan correctly implements the synthetic feature logic (e.g., Childhood Adversity Index) as defined in `PROJECT_REQUIREMENT.md`.
4. **Identify Edge Cases:** Look for data gaps (e.g., missing historical political data for new countries) and propose mitigations.
5. **Output a "Grill Report":**
   - **Critical Gaps:** Must-fix issues.
   - **DSA Suggestions:** Optimization opportunities.
   - **Edge Cases:** Potential failure points.
   - **Verdict:** [APPROVE] or [REVISE].

## Triggers
- Use this skill before starting any implementation task from the 7-step roadmap.
- Triggered by `/grill-plan`.
