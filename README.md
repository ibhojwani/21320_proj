# Industrial Makeup of US Counties: Impact over Time on Spread of COVID-19

A study of the affect on industry on COVID-19 by Athan Liu, Ishaan Bhojwani, and Joel Whittier

## Repository Overview

Our paper discussing our results can be found under 21320_Covid_Industry Writeup. Last revision June 11th 2020.

Code was written using Python. Recommended installation: Anaconda ver. 4.8.3, Python ver. 3.7.7.final.0

Python notebooks to observe data and verify data compilation and cleaning can be found under notebooks.

The scripts clean_data.py, regressions.py, and viz.py should be executed in a Python interpreter, and not via shell commands.

All 150 relevant regression tables can be found in regression_outputs. Condensed information can be found in summary_tables

## Recreating the research

The order of execution to recreate the experiment is clean_data.py, regressors.py, compiler.py, regressors.py, and viz.py. deaths_compiler.py is obsolete but remains in the repository to show a good faith effor by the researches to use deaths data as a metric and for others to verify the existence of problems with death data highlighted in the paper.