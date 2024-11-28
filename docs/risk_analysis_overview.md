# Risk Analysis Components

## 1. Risk Metrics
- **Severity (S)**: Impact if failure occurs (1-10)
  - 1-2: Minimal impact
  - 3-4: Minor impact
  - 5-6: Moderate impact
  - 7-8: Major impact
  - 9-10: Catastrophic impact

- **Likelihood (L)**: Probability of failure (1-10)
  - 1-2: Very unlikely
  - 3-4: Occasional
  - 5-6: Moderate probability
  - 7-8: Frequent
  - 9-10: Almost certain

- **Detectability (D)**: Ability to detect before impact (1-10)
  - 1-2: Almost certain to detect
  - 3-4: High chance to detect
  - 5-6: Moderate chance
  - 7-8: Low chance
  - 9-10: Almost impossible to detect

## 2. Risk Priority Number (RPN)
- RPN = Severity × Likelihood × Detectability
- Range: 1 to 1000
- Risk Levels:
  - Low: RPN < 100
  - Medium: 100 ≤ RPN < 200
  - High: RPN ≥ 200

## 3. Calculations Example
For a trade validation task:
- Severity = 7 (Major impact if validation fails)
- Likelihood = 3 (Occasional occurrence)
- Detectability = 4 (High chance to detect)
- RPN = 7 × 3 × 4 = 84 (Low risk)

## 4. Automatic Adjustments
- Gateway complexity increases severity
- Historical failures increase likelihood
- Monitoring capabilities improve detectability