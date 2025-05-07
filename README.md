# Product Availability and Taste Bias in Discrete Choice

**Authors**: René Leal-Vizcaíno, Juan José Merino & Edwin Muñoz-Rodriguez

## Introduction
This document provides instructions for executing the files in the repository associated with the paper "Product Availability and Taste Bias in Discrete Choice."

---

## File Naming Conventions
The repository contains code files with specific abbreviations at the start of their names. Below is a description of each:

| **Abbreviation** | **Description**                                                                 |
|------------------|-------------------------------------------------------------------------------|
| **BIAS**         | Distribution of estimated β's.                         |
| **POLOE**        | Probability of observing at least one exhaustion during the choice process.   |
| **PONAE**        | Probability of a number of alternatives exhausted as the choice unfolds.      |

---

## Steps to Execute the Files

### BIAS

1. **Run parallelization for simulations**:
   
   Run the `BIAS-parallelization.py` file to configure various parameters for the `BIAS-code.py` file. You can adjust the following settings:
   - `J` (products)
   - `q` (capacity)
   - Degree of product differentiation.
   - MC (Number of Monte Carlo simulations)
   - n_consumers (Number of consumers)
   - n_scripts (Number of scripts to run simultaneously)

3. **Generate bias plots**:

   Generate visualizations with the `BIAS-boxplots.R` file.
   
---

### POLOE

1. **Run parallelization for simulations**:

   Run the `POLOE-parallelization.py` file to configure various parameters for the `POLOE-code.py` file. You can adjust the following settings:
   - `J` (products)
   - `q` (capacity)
   - Degree of product differentiation.
   - MC (Number of Monte Carlo simulations)
   - n_consumers (Number of consumers)
   - n_scripts (Number of scripts to run simultaneously)
   
2. **Concatenate data**:
   
   After running the parallelization, combine all the generated data using the `POLOE-concatenate_data.py` file.
   
3. **Generate graphics**:
   
   Generate visualizations with the `POLOE-graphics.R` file.
   
---

### PONAE

1. **Run parallelization for simulations**:
   
   Run the `PONAE-parallelization.py` file to configure various parameters for the `PONAE-code.py` file. You can adjust the following settings:
   - `J` (products)
   - `q` (capacity)
   - Degree of product differentiation.
   - MC (Number of Monte Carlo simulations)
   - n_consumers (Number of consumers)
   - n_scripts (Number of scripts to run simultaneously)

3. **Generate graphics**:
   
   Generate visualizations with the `PONAE-graphics.R` file.
   
