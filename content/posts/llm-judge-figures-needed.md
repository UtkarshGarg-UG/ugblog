# LLM-as-a-Judge Blog: Figures to Create

This document lists all the figures/visualizations needed for the blog post, along with their descriptions and suggested visual treatments.

## Figure 1: The Development Loop
**Path:** `/llm-judge/development-loop.png`
**Location:** Part I - "The Development Loop That Actually Works"
**Description:** The iterative development loop for LLM judges
**Suggested visual:**
- Circular flowchart with 4 stages: Define → Analyze → Refine → Repeat
- Each stage with brief annotation
- Arrows showing iteration cycle
- Highlight "Read Analyses" as the key feedback mechanism

---

## Figure 2: Structured Output Schema
**Path:** `/llm-judge/structured-output-schema.png`
**Location:** Part II - "Why You Must Require Analysis Before Labels"
**Description:** Well-designed judge output schema showing field order
**Suggested visual:**
- JSON schema visualization or diagram
- Highlight the ordering: analysis → criterion_scores → label
- Visual indication of token generation order (left-to-right or top-to-bottom)
- Annotation showing "Reasoning first, verdict second"

---

## Figure 3: Binary Rubric Report Dashboard
**Path:** `/llm-judge/binary-rubric-report.png`
**Location:** Part II - "Designing the Rubric"
**Description:** Sample evaluation report for a binary rubric
**Suggested visual:**
- Bar chart showing per-criterion pass rates
- Example: Coverage 95%, Format 92%, Relevance 68%, Safety 98%
- Overall pass rate with CI: 65% [59%, 71%]
- Annotation highlighting that "Relevance" is the main failure mode

---

## Figure 4: Positional Bias Measurement
**Path:** `/llm-judge/positional-bias.png`
**Location:** Part III - "Positional Bias: The Pairwise Failure Mode"
**Description:** Position bias varies by model
**Suggested visual:**
- Bar chart showing % of times different models favor "first" position
- Example models: Model A (52%), Model B (63%), Model C (57%), Model D (69%)
- Horizontal reference line at 50% (no bias)
- Color-code bars by severity (green near 50%, yellow moderate, red high bias)

---

## Figure 5: Pairwise Randomization Flow
**Path:** `/llm-judge/pairwise-randomization.png`
**Location:** Part III - "Positional Bias" section
**Description:** The AB/BA randomization workflow
**Suggested visual:**
- Flowchart showing:
  1. Original pair (Output A, Output B)
  2. Random assignment (50% AB, 50% BA)
  3. Judge sees randomized order
  4. Verdict captured
  5. Remap to determine actual winner
- Include sample data showing the mapping

---

## Figure 6: Rankings with Confidence Intervals
**Path:** `/llm-judge/ranking-with-cis.png`
**Location:** Part III - "Aggregating Pairwise Wins"
**Description:** Model rankings with 95% bootstrap confidence intervals
**Suggested visual:**
- Horizontal bar chart with error bars
- Example: Model A: 1.45 [1.38, 1.52], Model B: 1.20 [1.15, 1.26], Model C: 1.18 [1.12, 1.24], Model D: 0.80 [0.74, 0.86]
- Highlight overlapping CIs between Model B and C
- Annotation: "No significant difference despite point estimate gap"

---

## Figure 7: Correlation Plot
**Path:** `/llm-judge/correlation-plot.png`
**Location:** Part IV - "Gold sets and human agreement"
**Description:** Judge scores vs. human scores with Pearson's r = 0.95
**Suggested visual:**
- Scatter plot: X-axis = Human scores (1-10), Y-axis = Judge scores (1-10)
- Best-fit line showing linear relationship
- Points slightly above the y=x diagonal (systematic inflation)
- Annotation showing r = 0.95

---

## Figure 8: Confusion Matrix
**Path:** `/llm-judge/confusion-matrix.png`
**Location:** Part IV - "Agreement Metrics"
**Description:** Confusion matrix for judge vs human labels
**Suggested visual:**
- 2x2 confusion matrix (Pass/Fail)
- Cells: Pass-Pass (40), Pass-Fail (10), Fail-Pass (5), Fail-Fail (45)
- Color intensity by count
- Annotations showing 85% agreement, κ = 0.70

---

## Figure 9: Bootstrap Distributions and CIs
**Path:** `/llm-judge/bootstrap-ci.png`
**Location:** Part IV - "Bootstrap confidence intervals"
**Description:** Bootstrap distributions for two systems
**Suggested visual:**
- Two overlapping distributions (System A and System B)
- System A centered at 75%, System B at 70%
- Show 95% CIs as shaded regions: [67%, 83%] and [62%, 78%]
- Highlight the overlap region
- Annotation: "Substantial overlap → difference may be noise"

---

## Figure 10: Self-Consistency Check
**Path:** `/llm-judge/self-consistency.png`
**Location:** Part IV - "Self-consistency checks"
**Description:** Self-consistency check across 5 runs
**Suggested visual:**
- Table or matrix showing:
  - Rows: 5 runs
  - Columns: Coverage, Format, Relevance, Label
  - Cells with checkmarks (✓) or X marks
  - Highlight Run 4 diverging on Relevance and final label
- Visual emphasis on the inconsistency

---

## Figure 11: A/A Test Dashboard
**Path:** `/llm-judge/aa-test-dashboard.png`
**Location:** Part VI - "Reliability Checks: A/A Tests"
**Description:** A/A test results dashboard
**Suggested visual:**
- Gauge or traffic light visualization
- Show metrics:
  - Pass-rate delta: 1.5pp (Green - within 2pp band)
  - Cohen's κ delta: 0.02 (Green - within 0.03 band)
- Color-coded zones: Green (good), Amber (review), Red (critical)
- Summary: "All metrics in green zone → Judge is stable"

---

## Figure 12: Slice Heatmap
**Path:** `/llm-judge/slice-heatmap.png`
**Location:** Part V - "Slice Your Results Aggressively"
**Description:** Heatmap of pass rates across slices
**Suggested visual:**
- 2D heatmap:
  - Rows: Topics (Technical, Creative, Factual)
  - Columns: Difficulty (Easy, Medium, Hard)
  - Cell colors: Green (high pass rate) → Red (low pass rate)
- Example values showing:
  - Overall: 75% (headline metric)
  - Creative/Easy: 90%
  - Technical/Hard: 60% (red cell - the hidden regression)
- Annotation highlighting the problematic cell

---

## Summary Statistics

- **Total figures added:** 12
- **Distribution:**
  - Part I: 1 figure
  - Part II: 2 figures
  - Part III: 3 figures
  - Part IV: 5 figures
  - Part V: 1 figure

## Notes for Creation

1. **Consistent style:** Use a consistent color palette and design language across all figures
2. **Accessibility:** Ensure sufficient color contrast and include text labels
3. **Format:** PNG format, optimized for web (recommend ~800-1200px width)
4. **Alt text:** Each figure already has a descriptive caption in the blog post
5. **Interactivity:** Consider making some figures interactive (SVG) for better user experience

## Existing Figures (Already in Post)

- **Intro iframe:** `/llm-judge/intro.html` (interactive demo)
- **Field order comparison:** `/llm-judge/field-order-results.png` (already exists, referenced from external source)
