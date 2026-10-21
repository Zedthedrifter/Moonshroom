# Lunar Gravitational and Photic Cues in *Pleurotus* Fruiting Phenology

Analysis code for a study examining whether lunar gravitational peaks and full moons influence the timing of peak harvest in cultivated *Pleurotus* species. The repository contains the data-processing pipeline, statistical models, and figure-generating scripts used in the manuscript.

---

## Overview

The study asks whether two environmental cues — lunar gravitational pull (LGP, defined by the change in lunar distance rate from negative to positive) and lunar phase (full moon, defined by the upper 1% of the lunar fraction) — disrupt the expected concentration of peak harvests on day 10 (Friday) of the fruiting cycle. Four species are analysed:

- *Pleurotus ostreatus*
- *Pleurotus pulmonarius*
- *Pleurotus saja-caju*

Three complementary analyses are implemented:

1. **Descriptive distribution of peak harvest days**, with boxplots and per-day summaries.
2. **Logistic regression** of the probability of a Friday peak on LGP or full-moon status, adjusting for year.
3. **Distributed Lag Non-Linear Models (DLNM)** of the continuous effect of gravitational force or lunar fraction on within-week normalized harvest, across lags 0–7.
4. **Uniformity tests** (chi-square and permutation) to confirm that LGP and full-moon days are not confounded with day-of-week.

---

