"""Copyright 2022 Alexey Akinshchikov
akinshchikov@gmail.com
"""

import os
import re
from typing import List, Set

import numpy as np
import pandas as pd

EPSILON: float = 1e-9


def batch_evaluate_match_problem(check_strings: np.ndarray,
                                 correct_answers: np.ndarray,
                                 answers2points: np.ndarray,
                                 ) -> np.ndarray:
    return np.fromiter(
        (evaluate_match_problem(check_string=check_string,
                                correct_answers=correct_answers,
                                answers2points=answers2points,
                                ) for check_string in check_strings),
        dtype=float,
    )


def batch_evaluate_sort_problem(check_strings: np.ndarray,
                                true_string: str) -> np.ndarray:
    return np.fromiter(
        (evaluate_sort_problem(check_string, true_string) for check_string in check_strings),
        dtype=int,
    )


def evaluate(check_dir: str = 'check',
             correct_dir: str = 'correct',
             output_dir: str = 'output',
             ) -> None:
    correct: np.ndarray = np.genfromtxt(fname=f'{correct_dir}/correct.csv',
                                        delimiter=',',
                                        dtype=str,
                                        )

    check: np.ndarray = np.genfromtxt(fname=f'{check_dir}/check.csv',
                                      delimiter=',',
                                      dtype=str,
                                      )

    problems_count: int = correct.shape[1]

    normalized_points_list: List[np.ndarray] = []

    for i in range(problems_count):
        if check[0, -problems_count + i] != correct[0, i]:
            raise ValueError('Problem indices or their order in "correct.csv" and "check.csv" are different.')

        check_strings: np.ndarray = check[1:, -problems_count + i]

        true_string: str = correct[1, i]

        max_normalized_points: float = float(correct[3, i])

        if correct[2, i] == 'SORT':
            raw_points: np.ndarray = batch_evaluate_sort_problem(check_strings=check_strings,
                                                                 true_string=true_string,
                                                                 )

            max_raw_points: float = evaluate_sort_problem(check_string=true_string,
                                                          true_string=true_string,
                                                          )

            min_raw_points: float = max_raw_points / 2
        elif correct[2, i] == 'MATCH':
            size: int = len(true_string)

            answers2points_path: str = f'../{correct_dir}/{correct[0, i]}_answers.csv'

            if os.path.exists(answers2points_path):
                raw_answers2points: np.ndarray = np.genfromtxt(fname=answers2points_path,
                                                               delimiter=',',
                                                               dtype=str,
                                                               )

                correct_answers: np.ndarray = raw_answers2points[:, 0]

                answers2points: np.ndarray = raw_answers2points[:, 1:].astype(float)
            else:
                correct_answers: np.ndarray = np.array(list(true_string))

                answers2points: np.ndarray = np.zeros([size, size])

                np.fill_diagonal(answers2points, 1.)

            raw_points: np.ndarray = batch_evaluate_match_problem(check_strings=check_strings,
                                                                  correct_answers=correct_answers,
                                                                  answers2points=answers2points,
                                                                  )

            max_raw_points: float = evaluate_match_problem(check_string=true_string,
                                                           correct_answers=correct_answers,
                                                           answers2points=answers2points,
                                                           )

            min_raw_points: float = 0.
        else:
            raise ValueError('Wrong problem type.')

        normalized_points: np.ndarray = (
                (raw_points - min_raw_points) / (max_raw_points - min_raw_points) * max_normalized_points + EPSILON
        ).astype(int)

        normalized_points = normalized_points.clip(min=0,
                                                   max=max_normalized_points,
                                                   ).astype(int)

        normalized_points_list.append(normalized_points)

    normalized_points_array: np.ndarray = np.stack(arrays=normalized_points_list,
                                                   axis=1,
                                                   )

    output: pd.DataFrame = pd.DataFrame(data=check[1:, :],
                                        columns=check[0, :],
                                        )

    output[check[0, -problems_count:]] = normalized_points_array

    output['Sum'] = output.sum(axis=1, numeric_only=True)

    output['Rank'] = output['Sum'].rank(ascending=False,
                                        method='min',
                                        ).astype(int)

    output.to_csv(path_or_buf=f'{output_dir}/output.csv',
                  index=False,
                  )


def evaluate_match_problem(check_string: str,
                           correct_answers: np.ndarray,
                           answers2points: np.ndarray,
                           ) -> float:
    size: int = correct_answers.shape[0]

    if len(check_string) != size:
        return 0.

    return sum(answers2points[i, j] if correct_answers[i] == check_string[j] else 0.
               for i in range(size) for j in range(size))


def evaluate_sort_problem(check_string: str,
                          true_string: str) -> float:
    check_set: List[str] = get_inversions_list(check_string)
    true_set: List[str] = get_inversions_list(true_string)

    return sum(pair in check_set for pair in true_set)


def get_inversions_list(string: str) -> List[str]:
    return sorted(list(get_inversions_set(string)))


def get_inversions_set(string: str) -> Set[str]:
    inversions: Set[str] = set()

    bracketless_string = re.sub(pattern='[\[\]]',
                                repl='',
                                string=string,
                                )

    for i in range(0, len(bracketless_string) - 1):
        for j in range(i + 1, len(bracketless_string)):
            inversions.add(bracketless_string[i] + bracketless_string[j])

    left_bracket: int = 0

    bracket_counter: int = 0

    for k in range(0, len(string)):
        if string[k] == '[':
            if bracket_counter != 0:
                raise ValueError(f'Wrong brackets usage in answer "{string}".')

            bracket_counter += 1

            left_bracket = k

        if string[k] == ']':
            if bracket_counter != 1:
                raise ValueError(f'Wrong brackets usage in answer "{string}".')

            bracket_counter -= 1

            for i in range(left_bracket + 1, k - 1):
                for j in range(i + 1, k):
                    inversions.remove(string[i] + string[j])

    return inversions
