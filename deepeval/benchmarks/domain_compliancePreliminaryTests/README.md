# Preliminary Benchmark Results: DomainComplianceMetric vs Plain GEval

## Setup

- **Judge model:** `llama-3.3-70b-versatile` via Groq
- **Test cases:** 12 manually designed cases
- **Domains:** Banking and healthcare
- **Threshold:** 0.7
- **Purpose:** Compare a generic GEval prompt against a domain-specific compliance metric on regulated-domain failure cases.

## Accuracy

| Metric | Accuracy |
|---|---:|
| Plain GEval generic prompt | 75% (9/12) |
| DomainComplianceMetric | 100% (12/12) |
| Improvement | +25 percentage points |

## Interpretation

These results should be interpreted as a small preliminary benchmark rather than a large-scale evaluation. The test set contains 12 targeted cases designed to check whether the metric catches domain-specific compliance failures that may be missed by a generic helpfulness/safety/accuracy prompt.

## Key Finding

`DomainComplianceMetric` performs better on borderline regulated-domain responses: outputs that appear helpful or factually plausible but violate domain-specific compliance expectations.

Examples include:

1. **Missing financial disclaimer**  
   A response may correctly mention historical investment performance but fail to state that past performance does not guarantee future results.

2. **No medical referral or safety hedge**  
   A response may provide medically relevant information but fail to advise consultation with a qualified healthcare professional.

3. **Implied diagnosis**  
   A response may infer a likely medical condition from partial symptoms or values without appropriate diagnostic caution.

## Why Plain GEval Misses Some Cases

The generic GEval baseline uses the broad criterion:

> Is this response accurate, helpful, and safe?

This can reward responses that are fluent, confident, and partially accurate, even when they miss domain-specific requirements such as financial disclaimers, medical escalation, or diagnostic uncertainty.

## Why DomainComplianceMetric Helps

`DomainComplianceMetric` adds domain-specific evaluation criteria. For regulated domains, this allows the evaluator to penalize responses that:

- make unsupported guarantees,
- provide unsafe medical or financial advice,
- omit necessary disclaimers,
- infer diagnoses or decisions beyond the provided context,
- fail to hedge appropriately when uncertainty exists.

## Limitation

This benchmark is intentionally small and targeted. The results demonstrate the usefulness of domain-specific evaluation criteria on selected compliance-sensitive cases, but they should not be interpreted as a broad statistical evaluation. A larger benchmark across more domains, models, and case distributions would be required for stronger empirical claims.

NOTE: This is just a preliminary test, not to claim its 100% accuracy. 