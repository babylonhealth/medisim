# Copyright 2020 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for creating similar term pairs"""

import os
import csv
import pandas as pd
from Levenshtein import distance as levenshtein_distance


def save_positive_instances(dataset_path,
                            easy_hard_split,
                            split_distance,
                            datasets,
                            datasets_easy,
                            dataset_names):

    for i, name in enumerate(dataset_names):

        file_name = os.path.join(dataset_path, name)
        if easy_hard_split:
            file_name_easy = file_name + "_easy_distance" + str(split_distance) + ".tsv"
            file_name = file_name + "_hard_distance" + str(split_distance) + ".tsv"
            datasets_easy[i].to_csv(file_name_easy, sep="\t", index=False, quoting=csv.QUOTE_NONE)
        else:
            file_name = file_name + ".tsv"

        datasets[i].to_csv(file_name, sep="\t", index=False, quoting=csv.QUOTE_NONE)

        # print statistics about new datasets
        if easy_hard_split:
            print(name + "_hard dataset has " +
                  str(datasets[i].shape[0]) +
                  " positive instances")
            print(name + "_easy dataset has " +
                  str(datasets_easy[i].shape[0]) +
                  " positive instances")
        else:
            print(name + " dataset has " +
                  str(datasets[i].shape[0]) +
                  " positive instances")



def create_dataframes_without_duplicates(zipped_datasets, easy_hard_split, lab1, lab2):

    datasets = []
    datasets_easy = []

    for dataset, dataset_easy in zipped_datasets:
        dataset_dataframe = pd.DataFrame(dataset)
        dataset_easy_dataframe = pd.DataFrame(dataset_easy)

        remove_duplicates(dataset_dataframe, lab1, lab2)
        if easy_hard_split:
            remove_duplicates(dataset_easy_dataframe, lab1, lab2)

        datasets.append(dataset_dataframe)
        datasets_easy.append(dataset_easy_dataframe)

    return datasets, datasets_easy


def remove_duplicates(dataframe, lab1, lab2):
    deletion_list = []
    # remove normal duplicates
    dataframe.drop_duplicates(inplace=True)

    # create table with column labels switched
    dataframe_switch = dataframe.copy()
    dataframe_switch.rename({lab1 : lab2, lab2 : lab1}, axis=1, inplace=True)

    # find all rows which are reverse duplicates
    # (each pair will occur as lab1 - lab2 and lab2, lab1)
    # by merging the original and the reversed tables on the source and target labels
    # since by default the merge mode is 'inner',
    # the merged table only contains lab1-lab2 pairs that occur in
    # both tables (all others are ignored)
    # --> the merged table only contains reverse duplicates
    # 1 instance for each duplicate
    # - if a table contains the same lab1-lab2 twice, then 2 instances in the new table
    dataframe_merge = \
        dataframe.merge(dataframe_switch, on=[lab1, lab2], right_index=True)
    index_list = dataframe_merge.index.values.tolist()

    # go through all indices of the original table that have reverse duplicates
    # (i.e. all indices in the merged table)
    while index_list != []:

        # concept pair that has a reverse duplicate
        l1 = dataframe_merge.loc[index_list[0], lab1]
        l2 = dataframe_merge.loc[index_list[0], lab2]

        # index of reverse duplicate
        # (should have exactly one element, but for generality this can be applied to multiple ones)
        to_delete = dataframe_merge[(dataframe_merge[lab2] == l1)
                                    & (dataframe_merge[lab1] == l2)].index.values.tolist()

        # add index to rows to be deleted and
        # delete it from the indexes to be further investigated
        # (each pair should only be investigated once,
        # but reverse duplicates of course all occur twice)
        for i in to_delete:
            index_list.remove(i)
            deletion_list.append(i)
        index_list.pop(0)

    # delete all reverse duplicates from the table
    dataframe.drop(deletion_list, inplace=True)


def create_term_pairs(pairs,
                      easy_hard_split,
                      split_distance,
                      label1_name,
                      label2_name,
                      pairs_set,
                      pairs_set_easy):

    for lab1, lab2 in pairs:
        if lab1.lower() == lab2.lower():
            continue

        # check if Levenstein distance between the two labels
        # is smaller or equal to the max distance defined
        # if a dataset split into easy/hard is desired
        if easy_hard_split and levenshtein_distance(lab1.lower(), lab2.lower()) <= split_distance:
            pairs_set_easy[label1_name].append(lab1)
            pairs_set_easy[label2_name].append(lab2)
        else:
            pairs_set[label1_name].append(lab1)
            pairs_set[label2_name].append(lab2)
    return pairs_set, pairs_set_easy


# many pref labels end with a parenthesis indicating its semantic type
def clean_pref_term(pref_label):
    pref_label_cleaned = pref_label

    if pref_label.endswith(")"):
        split_pref_label = pref_label.rsplit(" (", 1)
        # check if the parenthesis at the end is something other than a semantic type, e.g.
        # (...) or (...) ; ... & (...) ; ... (& ...)
        # some strange labels end with a parenthesis but have no starting one, e.g.
        # Lactation: [problems] or [& obstetric breast disorders NOS])
        if len(split_pref_label) > 1 and not split_pref_label[0].endswith((" or", "&")) and \
                not split_pref_label[1].startswith("&"):
            pref_label_cleaned = split_pref_label[0]
    return pref_label_cleaned
