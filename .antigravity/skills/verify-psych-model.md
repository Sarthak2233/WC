# Skill: Verify Psych Model
Name: verify-psych-model
Description: Validates the statistical integrity and causal plausibility of synthetic psychopolitical features.

## Instructions
When this skill is activated, you must:
1. **Check Distribution:** Ensure synthetic features (e.g., Psyche Score, PPI) are properly normalized and follow expected distributions.
2. **Causal Audit (Causal Pattern Library):** Verify the heuristic logic against the 10 core patterns:
   - **Host-Nation Surge:** Positive PPI boost for hosts.
   - **Legacy Pressure:** Performance penalty for high-expectation former winners.
   - **Defending-Champion Slump:** Adversity penalty for holders.
   - **High-Tension Nationalism:** Semi-final peaks vs. final collapses in autocratic regimes.
   - **Crisis-to-Motivation:** 20% boost for high-adversity squads in post-crisis years.
   - **Collectivist Calm vs. Individualist Panic:** UAI modulation of pressure response.
   - **Internal Division:** Cohesion penalty for ethnically/politically divided teams.
   - **Past Glory Guilt:** Weight of historical victory on young players.
   - **Youth vs. Experience:** Performance delta between veteran squads and youth.
   - **Geopolitical Shock:** Impact of external wars/scandals during tournaments.
3. **Data Integrity:** Check for look-ahead bias in historical feature computation.
4. **Output Verification Report:**
   - **Statistical Summary:** Mean, std, outliers.
   - **Plausibility Check:** Do the scores make "human sense" (e.g., 2018 Croatia having a high adversity score)?
   - **Bias Alert:** Identify potential sources of systemic bias.

## Triggers
- Use this skill after generating any new synthetic feature set.
- Triggered by `/verify-psych-model`.
