# Evaluation audit

The recovered scorer reproduces author-local matching, not the official BFCL checker.

1. **Architecture:** one function head and six positional argument heads represent one call. Properties retain insertion order. Heads beyond the chosen schema arity have no parameter semantics and are ignored in both scorers.
2. **Mobile Actions:** 322/1,283 rows contain multiple GT entries. These are sequential conversions with supplied reference history. First rows target the first call; subsequent rows target the final call. Requiring all GT calls at once was an audit mistake: the earlier 66.25% exact-sequence diagnostic is invalid for this protocol. Historical 84.72% and strict 82.54% use one-call evaluation. Neither is whole-trajectory success.
3. **Extra arguments:** historical scoring ignores keys absent from GT. Strict scoring rejects those keys after schema conversion. This does not mean unused raw heads become parameters. GT itself may omit schema-valid optional arguments; the strict rule is a conservative local diagnostic, not an official semantic checker.
4. **Capacity selection:** any candidate tool with more than six properties excludes the row. This changes the tested population and may affect comparability. 5,437/5,647 rows score; exclusions are recorded in the manifest.
5. **Ground truth issues:** one malformed exec GT line and one Live ID mismatch are preserved and explicitly excluded. No silent repair.
6. **Metric boundary:** exec subsets use local acceptable-value matching, not program execution. BFCL coverage is the listed simple/multiple subsets, not the entire official benchmark. `multiple` describes multiple candidate tools and must not automatically be read as parallel output calls.
7. **Integrity:** full scoring requires every dataset and input ID exactly once. Missing/unknown/duplicate predictions must fail. Data hashes are verified before scoring. Published raw inference and historical run hashes are retained.

Strict scores: Non-Live 92.67%, Live 76.17%, Exec 87.25%, Mobile Actions 82.54%, Others 86.03%. Historical scores: 92.67%, 76.17%, 89.93%, 84.72%, 86.60%. Group scores pool correct/scored counts.

Close agreement with the paper is evidence of local reproduction, not proof of no contamination, no conversion bias, or official leaderboard equivalence. v2 remains unevaluated. Official comparisons require a pinned upstream dataset/checker and explicit handling of unsupported cases.
