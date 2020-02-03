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

"""Creation of dissimilar term pairs"""

import random
import os
import statistics
import csv
import operator
from collections import OrderedDict
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
import pandas as pd


def is_existing_pair(existing_pairs, label1, label2):
    return (not existing_pairs.loc[
        (existing_pairs['source'] == label1) & (existing_pairs['target'] == label2)].empty or \
    not existing_pairs.loc[
        (existing_pairs['target'] == label1) & (existing_pairs['source'] == label2)].empty)


# write some statistics about the negative instances (mean, max, min Levenshtein distance)
def write_statistics_to_file(statistics_filename,
                             distances,
                             no_of_positive_instances,
                             dataset_name):
    with open(statistics_filename, 'a') as stats:
        stats.write(dataset_name + "\n")
        stats.write("Number of positive instances: " + str(no_of_positive_instances) + "\n")
        stats.write("Mean Levenshtein Distance: " + str(statistics.mean(distances)) + "\n")
        stats.write("Median Levenshtein Distance: " + str(statistics.median(distances)) + "\n")
        stats.write("Max Levenshtein Distance: " + str(max(distances)) + "\n")
        stats.write("Min Levenshtein Distance: " + str(min(distances)) + "\n")
        stats.write("\n")


##################################################################
# Random strategy for negative sampling
##################################################################
def create_random_pairs(positive_instances,
                        positive_pairs_all_datasets,
                        existing_negatives):

    random.seed(42)
    # holds the Levenshtein distance of each concept pair
    distances = []

    # tracks already created negative pairs as tuples, i.e. (l1,l2), to avoid duplicate creation
    new_negative_pairs = []

    for i, row in tqdm(positive_instances.iterrows(), total=positive_instances.shape[0]):
        label1 = row['source']

        # initialise random index
        random_index = i

        # make sure that no term pair duplicates or reverse duplicates are created
        # comparing to both positive and negative concept pairs
        while random_index == i or\
            is_existing_pair(positive_pairs_all_datasets, label1, label2) or\
            is_existing_pair(existing_negatives, label1, label2) or\
            (label1, label2) in new_negative_pairs or (label2, label1) in new_negative_pairs\
            or label1.lower() == label2.lower():

            # choose a new random index and source vs target and get a new pairing term

            random_index = random.randint(0, positive_instances.shape[0]-1)
            source_or_target = random.choice(['source', 'target'])
            label2 = positive_instances.loc[random_index][source_or_target]

        distances.append(levenshtein_distance(label1.lower(), label2.lower()))
        new_negative_pairs.append((label1, label2))

    return new_negative_pairs, distances




##################################################################
# Levenshtein strategy for negative sampling
##################################################################

def create_minimal_distance_pairs(positive_instances,
                                  positive_pairs_all_datasets,
                                  existing_negatives):
    random.seed(42)

    # holds the Levenshtein distance of each concept pair
    distances = []

    # tracks already created negative pairs as tuples, i.e. (l1,l2), to avoid duplicate creation
    new_negative_pairs = []

    # find all instances of each source concept
    unique_source_concepts = positive_instances.groupby('source')

    # for each concept, create a list of usable concepts that are not positive similarity instances
    # and choose the ones with smallest Levenshtein distance as a difficult negative sample
    for label1, group in tqdm(unique_source_concepts, total=unique_source_concepts.ngroups):

        possible_targets = get_possible_targets(group, new_negative_pairs, positive_instances)
        distances_possible_terms, possible_targets = \
            get_levenshtein_possible_targets(possible_targets, label1)

        # find the N minimal distances (for N positive pairs of the concept)
        # and the respective pairing concept with this minimal distance
        sorted_targets_and_distances = \
            [(label, d) for d, label in sorted(zip(distances_possible_terms, possible_targets),
                                               key=operator.itemgetter(0))]

        min_dist_tuples = []
        for i in range(0, len(group)):

            # get the smallest Levenshtein distance
            if not min_dist_tuples:
                min_dist_tuples, sorted_targets_and_distances = \
                    get_min_distance_tuples(sorted_targets_and_distances)

            # choose a random term with minimal distance
            label2, distance = min_dist_tuples.pop(random.randint(0, len(min_dist_tuples)-1))

            while is_existing_pair(positive_pairs_all_datasets, label1, label2) or \
            is_existing_pair(existing_negatives, label1, label2):

                if not min_dist_tuples:
                    min_dist_tuples, sorted_targets_and_distances = \
                        get_min_distance_tuples(sorted_targets_and_distances)

                label2, distance = min_dist_tuples.pop(random.randint(0, len(min_dist_tuples) - 1))

            new_negative_pairs.append((label1, label2))
            distances.append(distance)

    return new_negative_pairs, distances


def get_min_distance_tuples(sorted_targets_and_distances):
    min_dist_tuples = []
    min_label, min_distance = sorted_targets_and_distances.pop(0)
    min_dist_tuples.append((min_label, min_distance))

    # find all terms with the same minimal dinstance
    while sorted_targets_and_distances[0][1] == min_distance:
        min_dist_tuples.append(sorted_targets_and_distances.pop(0))

    return min_dist_tuples, sorted_targets_and_distances


def get_possible_targets(group, new_negative_pairs, positive_instances):

    # exclude the similarity pairs of this concept from table to be used to create negative pair
    usable_labels = positive_instances.drop(group.index)

    # all targets of the current concept are synonyms
    # that should not be paired with the current concept,
    # so is of course the current concept itself
    synonyms = group['target'].tolist()
    label1 = positive_instances.loc[group.index.tolist()[0], 'source']
    synonyms.append(label1)

    # find all concepts that are paired with the synonyms (as source or target)
    concepts_to_exclude = \
        usable_labels[usable_labels.target.isin(synonyms)]['source'].tolist()
    concepts_to_exclude = \
        concepts_to_exclude + usable_labels[usable_labels.source.isin(synonyms)]['target'].tolist()

    # exclude all concept pairs containing a concept that's also paired with a synonym
    usable_labels = usable_labels[
        ~usable_labels.source.isin(concepts_to_exclude)]
    usable_labels = usable_labels[~usable_labels.target.isin(concepts_to_exclude)]

    # the sources and targets of the remaining pairs can be paired with the current concept
    usable_list = \
        usable_labels['source'].unique().tolist() + usable_labels['target'].unique().tolist()
    usable_list = list(OrderedDict.fromkeys(usable_list))

    # make sure no reverse duplicates are created,
    # i.e. if (X, lab1) already occurs in the negative instances,
    # exlude X - note that (lab1, X) won't occur in the neg samples
    # since same concepts are handled together
    labels_from_existing_negative_instances = \
        [lab for (lab, l) in new_negative_pairs if l == label1]
    usable_list_final = \
        [x for x in usable_list if x not in labels_from_existing_negative_instances]

    return usable_list_final


# for each potential pairing of terms, compute their Levenshtein distance and store it in a list
# record labels that have Levenshtein distance 0 (i.e. only the casing of the concepts is different)
# to exlcude them later
def get_levenshtein_possible_targets(possible_targets, label1):

    distances_possible_terms = []
    distance0 = []

    for i, label2 in enumerate(possible_targets):
        d = levenshtein_distance(label2.lower(), label1.lower())
        if d == 0:
            distance0.append(i)
        else:
            distances_possible_terms.append(d)

    new_possible_targets = [x for i, x in enumerate(possible_targets) if i not in distance0]

    return distances_possible_terms, new_possible_targets


##################################################################

def negative_sampling(strategy,
                      full_new_dataset_path,
                      positive_instances,
                      statistics_path,
                      positive_pairs_all_datasets,
                      existing_negatives):

    # create negative instances according to chosen strategy
    if strategy == 'simple':
        new_negative_pairs, distances =\
            create_random_pairs(positive_instances, positive_pairs_all_datasets, existing_negatives)

    elif strategy == 'advanced':
        new_negative_pairs, distances = \
            create_minimal_distance_pairs(positive_instances,
                                          positive_pairs_all_datasets,
                                          existing_negatives)
    else:
        raise Exception('Unknown negative sampling strategy chosen!')

    # positive instances
    positive_pairs_with_scores = []
    for i, row in positive_instances.iterrows():
        positive_pairs_with_scores.append(row['source'] + "\t" + row['target'] + "\t1\n")

    # negative instances
    new_negative_pairs_with_scores = \
        [label1 + "\t" + label2 + "\t0\n" for (label1, label2) in new_negative_pairs]

    new_dataset_with_scores = positive_pairs_with_scores + new_negative_pairs_with_scores
    random.shuffle(new_dataset_with_scores)

    # save newly created dataset
    with open(full_new_dataset_path + '_' + strategy + '.txt', "w") as output:
        output.writelines(new_dataset_with_scores)

    # save statistics about new negative instances
    write_statistics_to_file(statistics_path + '_' + strategy + '.txt',
                             distances, positive_instances.shape[0],
                             full_new_dataset_path + '_' + strategy)

    return new_negative_pairs


def read_existing_positive_instances(positive_instance_datasets, dataset_path):

    # get all positive instances
    li = []
    for f in positive_instance_datasets:
        # all positive instances in the FSN_SYN datasets are also in the SYN_SYN datasets,
        # so no need to load them
        if "FSN_SYN" in f or f.startswith('._'):
            continue
        df = pd.read_csv(os.path.join(dataset_path, f), sep="\t",
                         quoting=csv.QUOTE_NONE, keep_default_na=False,
                         header=0, names=['source', 'target'])
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


##################################################################
# MAIN
##################################################################

def negative_instances(dataset_path, strategies):

    # path to save statistics
    statistics_path = dataset_path + "negative_sampling_statistics"

    # ORDER MATTERS!
    positive_instance_datasets = [
        'possibly_equivalent_to_easy_distance5.tsv',
        'possibly_equivalent_to_hard_distance5.tsv',
        'replaced_by_easy_distance5.tsv',
        'replaced_by_hard_distance5.tsv',
        'same_as_easy_distance5.tsv',
        'same_as_hard_distance5.tsv',
        'FSN_SYN_easy_distance5.tsv',
        'FSN_SYN_hard_distance5.tsv',
        'SYN_SYN_easy_distance5.tsv',
        'SYN_SYN_hard_distance5.tsv'
    ]

    positive_pairs_all_datasets = read_existing_positive_instances(positive_instance_datasets,
                                                                   dataset_path)

    # consider the random and advanced strategy separately
    # as negative instances are considered separately
    for strategy in strategies:

        # dataframes to keep track of already created negative instances (to prevent duplicates)
        existing_negatives_to_consider = \
            pd.DataFrame(columns=['source', 'target', 'trueScore'])
        existing_negatives_from_substitution = \
            pd.DataFrame(columns=['source', 'target', 'trueScore'])
        existing_negatives_SYN_SYN = \
            pd.DataFrame(columns=['source', 'target', 'trueScore'])

        for positive_dataset in positive_instance_datasets:

            print(positive_dataset)
            new_dataset_name = dataset_path + positive_dataset.rsplit(".", 1)[0] + "_with_neg"

            # read the positive instances into a dataframe
            positive_instances = pd.read_csv(os.path.join(dataset_path, positive_dataset),
                                             sep="\t",
                                             quoting=csv.QUOTE_NONE,
                                             keep_default_na=False,
                                             header=0,
                                             names=['source', 'target'])

            # create negative instances for this dataset
            new_negative_pairs = negative_sampling(strategy,
                                                   new_dataset_name,
                                                   positive_instances,
                                                   statistics_path,
                                                   positive_pairs_all_datasets,
                                                   existing_negatives_to_consider)

            # turn these negative instances into a dataframe
            new_negatives = pd.DataFrame(new_negative_pairs, columns=['source', 'target'])
            new_negatives['trueScore'] = 0

            # substitution datasets are processed first,
            # so existing negative pairs are only those constructed
            # in the other substitution datasets
            if not 'SYN' in positive_dataset:
                existing_negatives_from_substitution = \
                    pd.concat([existing_negatives_from_substitution, new_negatives],
                              axis=0, ignore_index=True)
                existing_negatives_to_consider = existing_negatives_from_substitution

            # FSN_SYN are processed second,
            # existing negative pairs are from substitutions plus FSN_SYN so far
            elif 'FSN_SYN' in positive_dataset:
                existing_negatives_to_consider = \
                    pd.concat([existing_negatives_to_consider, new_negatives],
                              axis=0, ignore_index=True)

            # all datasets are processed third, here prefToAlt negatives are not considered,
            # so only deletion and negatives in all datasets so far
            elif 'SYN_SYN' in positive_dataset:
                existing_negatives_SYN_SYN = pd.concat([existing_negatives_SYN_SYN, new_negatives],
                                                       axis=0, ignore_index=True)
                existing_negatives_to_consider = \
                    pd.concat([existing_negatives_from_substitution, existing_negatives_SYN_SYN],
                              axis=0, ignore_index=True)

            else:
                raise Exception('unknown dataset %s' % positive_dataset)
