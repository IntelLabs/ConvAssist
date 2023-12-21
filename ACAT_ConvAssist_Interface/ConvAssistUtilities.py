"""
 Copyright (C) <year(s)> Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""

"""
Imports
"""


def sort_List(prediction_list, amount_predictions):
    """
    Gets a specific amount of predictions in descending order

    :param prediction_list: original predictions list
    :param amount_predictions: amount of predictions requested
    :return: Sorted predictions list
    """
    new_convAssist_list = []
    try:
        temp_convAssist_list = sorted(prediction_list, key=lambda x: (x[1]), reverse=True)
        for x in range(amount_predictions):
            if x >= len(prediction_list):
                break
            element_list = temp_convAssist_list[x]
            new_convAssist_list.append(element_list)
    except Exception:
        new_convAssist_list = []
    return new_convAssist_list
