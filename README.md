# How to use?

To use `test_evaluator` run `python main.py` or import function `evaluate` from `functions` to your code.

# Problem types

`test_evaluator` evaluates the test with 2 types of problems:
 
* `MATCH`

* `SORT`

# Files description

* `check/check.csv` is a file with test answers for the evaluation.

* `correct/correct.csv` is a file with correct answers.
The first row is problem labels.
The second row is problem types.
The third row is problem maximum points.\
Use `[]` around letters corresponding to equal values for `SORT` problems.

* `correct/<problem index>.csv` is an optional file with custom points table for `MATCH` problems.

* `output/output.csv` is a file with results.
