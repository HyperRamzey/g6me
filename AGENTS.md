1.The Sabotage Rule:
"Whenever you write a unit test for a non-trivial invariant (like a ramp, a guard band, or a parser check), you must verify the test discriminates by temporarily sabotaging the feature (or running a 5-line python check) to confirm the test fails without it."

2.The No-Prose-Algebra Rule:
"Never derive ring-buffer index inequalities, filter delay offsets, or floating-point convergence bounds in prose. Always run a quick python3 -c one-liner to check the exact numbers before editing code."

3.The Anomaly-First Rule:
"If a benchmark, counter, or test output looks surprising (e.g., a limiter engaging on >50% of frames when only 0.8 dB of peaks exceed the threshold), assume a bug in the measurement/code first; never rationalize surprising numbers into documentation until verified."
