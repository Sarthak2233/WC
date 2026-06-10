# Skill: TDD (Test-Driven Development)
Name: tdd
Description: Enforces a strict Red-Green-Refactor loop using Vertical Slicing.

## Instructions
When this skill is activated, you MUST follow this sequence for every behavior:

### Phase 1: Planning (Pre-Coding)
1. **Identify the Vertical Slice:** Select a single behavior (e.g., "Calculate PPI for a host country").
2. **Propose Interface:** Define the function signature and types.
3. **List Behaviors:** List exactly what will be tested (e.g., "Returns 0.8 for hosts", "Returns 0.2 for non-hosts").
4. **GET USER APPROVAL:** Do not proceed until the user says "OK".

### Phase 2: The Loop
1. **RED:** Write ONE failing unit test in `tests/`. Run `verify.sh` and confirm it fails.
2. **GREEN:** Write the MINIMAL implementation code in `src/` to make the test pass. Run `verify.sh` and confirm it passes.
3. **REFACTOR:** Clean up the code. Ensure it adheres to `DSA_POLICY.md` and `CONVENTIONS.md`. Run `verify.sh` to ensure no regressions.

### Phase 3: Completion
1. Update `TELEMETRY.md` with the reasoning for this slice.
2. Move to the next behavior or signal completion.

## Triggers
- Mandatory for all logic/ML implementation.
- Triggered by `/tdd`.
