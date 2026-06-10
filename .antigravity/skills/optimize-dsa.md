# Skill: Optimize DSA
Name: optimize-dsa
Description: Refactors code to improve time and space complexity according to the DSA_POLICY.md.

## Instructions
When this skill is activated, you must:
1. **Profile the Target Code:** Identify the most computationally expensive sections (loops, joins, recursions).
2. **Apply Policy Rules:**
   - Replace loops with vectorized Pandas/NumPy operations.
   - Use generators for large dataset processing.
   - Implement memoization for repeated calculations.
3. **Validate Performance:** Estimate the new complexity and compare it to the old one.
4. **Unit Test:** Ensure the refactored code maintains behavioral correctness.

## Triggers
- Triggered when a performance bottleneck is identified.
- Triggered by `/optimize-dsa`.
