# Ponytail policy for TwinLens

Use Ponytail `full`: understand the end-to-end flow, then stop at the first rung that works.

1. Do not build speculative features.
2. Reuse existing code before adding code.
3. Prefer Python stdlib, browser-native features, SQLite constraints, and installed dependencies.
4. Add no abstraction, dependency, service, or file without a present requirement.
5. Keep security, trust-boundary validation, accessibility, calibration knobs, and data-loss handling intact.
6. Mark deliberate ceilings with `ponytail:` and state the upgrade trigger.
7. Non-trivial logic needs one runnable check; no test framework unless required.

Derived from DietrichGebert/ponytail (MIT): https://github.com/DietrichGebert/ponytail
