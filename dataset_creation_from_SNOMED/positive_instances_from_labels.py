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

"""Creation of similar term pairs from SNOMED concept labels"""

import itertools
import csv
import os
import pandas as pd
from tqdm import tqdm

from dataset_creation_from_SNOMED.snomed_id import SnomedID
from dataset_creation_from_SNOMED.positive_instances_utils import save_positive_instances
from dataset_creation_from_SNOMED.positive_instances_utils import create_dataframes_without_duplicates
from dataset_creation_from_SNOMED.positive_instances_utils import create_term_pairs
from dataset_creation_from_SNOMED.positive_instances_utils import clean_pref_term


def _label_sanity_check(alt, pref, concept):
    if not isinstance(alt, str):
        raise Exception("Alt label not a string: %s \n type is: %s \n concept ID: %s"
                        % (alt, type(alt), concept))
    if not isinstance(pref, str):
        raise Exception("Pref label not a string: %s \n type is: %s \n concept ID: %s"
                        % (alt, type(alt), concept))
    if pref.strip() == "":
        raise Exception("Empty string pref label of conept ID %s" % (concept))
    if alt.strip() == "":
        raise Exception("Empty string alt label of concept ID %s" % (concept))
    if alt.lower() == pref.lower():
        raise Exception('Pref and alt label are the same despite previous filtering \n pref: %s \n alt: %s' \
                        % (pref, alt))


def label_id_sanity_check(label_id_alt_entries,
                          label_id_pref_entries,
                          label_id_alt_terms,
                          label_id_pref_terms,
                          label_id):
    if label_id_alt_entries.shape[0] > 1:
        raise Exception('More than 1 entry for max year (alt label %s ) \n %s'
                        % (label_id, label_id_alt_entries))
    if label_id_pref_entries.shape[0] > 1:
        raise Exception('More than 1 entry for max year (pref label %s) \n %s'
                        % (label_id, label_id_pref_entries))
    if len(label_id_alt_terms) > 1:
        raise Exception("Multiple terms for the same alt label ID %s: \n %s \n %s" \
                        % (label_id, label_id_alt_entries, label_id_alt_terms))
    if len(label_id_pref_terms) > 1:
        raise Exception("Multiple terms for the same pref label ID %s: \n %s \n %s" \
                        % (label_id, label_id_pref_entries, label_id_pref_terms))
    if len(label_id_pref_terms) > 0 and len(label_id_alt_terms) > 0:
        raise Exception("Label ID %s is both pref and alt \n pref: %s \n alt: %s" \
                        % (label_id, label_id_pref_entries, label_id_alt_entries))
    if len(label_id_pref_terms) == 0 and len(label_id_alt_terms) == 0:
        raise Exception("Label ID %s is neither pref nor alt \n pref: %s \n alt: %s" \
                        % (label_id, label_id_pref_entries, label_id_alt_entries))


def check_exactly_one_pref_label(pref_list, concept):
    if len(pref_list) > 1:
        raise Exception("More than one pref label for concept %s: \n %s" %(concept, pref_list))
    if len(pref_list) == 0:
        raise Exception("No pref label for concept %s \n %s" %(concept, pref_list))


def label_sanity_check(label_pairs, concept):
    for lab1, lab2 in label_pairs:
        _label_sanity_check(lab1, lab2, concept)


def is_active_and_medical_concept(concept, concepts):
    concept_entries = concepts[concepts["id"] == concept]
    return is_active_concept(concept_entries) and is_medical_concept(concept_entries)


# check if the concept is still active, if not, don't include this concept in dataset
# (descriptions may be active even if concept is inactive! they are not de-activated!)
def is_active_concept(concept_entries):
    return concept_entries.loc[concept_entries["effectiveTime"].idxmax()].active


# don't use concepts which are in the model component module (mainly relationships and
# descriptional concepts like 'Inactive Value')
def is_medical_concept(concept_entries):
    return not concept_entries.loc[concept_entries["effectiveTime"].idxmax()].moduleId\
               == SnomedID.MODEL_COMPONENT_MODULE.value


def get_pref_and_alt_labels(labels, concept):
    # get all labels and their IDs for the concept
    concept_labels = labels[labels['conceptId'] == concept]
    concept_label_dict = {'pref': [], 'alt': []}

    # extract all current labels of this concept and split them into pref and alt
    for label_id in concept_labels.id.unique():

        # all entries for this label ID
        label_id_entries = concept_labels[concept_labels["id"] == label_id]

        # process a label if it's active
        if label_id_entries.loc[label_id_entries["effectiveTime"].idxmax()].active:
            # split the entries of a label into those regarding pref and those regarding alt labels
            # using them only if they are most recent entries
            # a label can also change from pref to alt label or vice versa,
            # here we only take the most recent one
            # for a given label we usually expect that one of the below is empty
            # (i.e. a label is either pref or alt)
            label_id_alt_entries = label_id_entries[
                (label_id_entries["effectiveTime"] == label_id_entries["effectiveTime"].max()) &
                (label_id_entries['typeId'] == SnomedID.SYNONYM_DESCRIPTION.value)]
            label_id_pref_entries = label_id_entries[
                (label_id_entries["effectiveTime"] == label_id_entries["effectiveTime"].max())
                & (label_id_entries['typeId'] == SnomedID.FSN_DESCRIPTION.value)]

            # initialise pref or alt terms with the current label
            # one of them should have one entry, the other none
            label_id_alt_terms = label_id_alt_entries.term.tolist()
            label_id_pref_terms = label_id_pref_entries.term.tolist()

            label_id_sanity_check(label_id_alt_entries,
                                  label_id_pref_entries,
                                  label_id_alt_terms,
                                  label_id_pref_terms,
                                  label_id)

            # add label_id to alt or pref labels of currently processed concept
            concept_label_dict['alt'].extend(label_id_alt_terms)
            concept_label_dict['pref'].extend(label_id_pref_terms)

    check_exactly_one_pref_label(concept_label_dict['pref'], concept)

    return concept_label_dict



##################################################################
# MAIN
##################################################################

def positive_instances_from_labels(easy_hard_split,
                                   split_distance,
                                   snomed_path,
                                   dataset_path):
    # input SNOMED files
    labels = pd.read_csv(os.path.join(snomed_path, "sct2_Description_Full-en_INT_20190131.txt"),
                         sep="\t", header=0,
                         quoting=csv.QUOTE_NONE, keep_default_na=False)
    concepts = pd.read_csv(os.path.join(snomed_path, "sct2_Concept_Full_INT_20190131.txt"),
                           sep="\t", header=0,
                           quoting=csv.QUOTE_NONE, keep_default_na=False)

    fsn_syn = {'pref':[], 'alt':[]}
    fsn_syn_easy = {'pref':[], 'alt':[]}
    syn_syn = {'label1':[], 'label2':[]}
    syn_syn_easy = {'label1':[], 'label2':[]}

    # create fsn-syn and syn-syn label pairs for all concepts (concept IDs)
    for concept in tqdm(concepts.id.unique()):

        if not is_active_and_medical_concept(concept, concepts):
            continue

        # extract all current labels of this concept and split them into pref and alt
        concept_label_dict = get_pref_and_alt_labels(labels, concept)

        pref_label = clean_pref_term(concept_label_dict['pref'][0])

        concept_label_dict['alt'] =\
            [l for l in concept_label_dict['alt'] if l.lower() != pref_label.lower()]

        label_sanity_check(itertools.product(concept_label_dict['alt'], [pref_label]), concept)

        # construct fsn-syn positive instances
        fsn_syn_label_pairs = itertools.product(concept_label_dict['alt'], [pref_label])
        fsn_syn, fsn_syn_easy = create_term_pairs(fsn_syn_label_pairs,
                                                  easy_hard_split,
                                                  split_distance,
                                                  'alt',
                                                  'pref',
                                                  fsn_syn,
                                                  fsn_syn_easy)
        
        # add pref label to the other alt labels to create syn-syn instances
        concept_label_dict['alt'].insert(0, pref_label)

        # construct syn-syn positive instances
        syn_syn_label_pairs = itertools.combinations(concept_label_dict['alt'], 2)
        syn_syn, syn_syn_easy = create_term_pairs(syn_syn_label_pairs,
                                                  easy_hard_split,
                                                  split_distance,
                                                  'label1',
                                                  'label2',
                                                  syn_syn,
                                                  syn_syn_easy)

    [syn_syn_dataframe], [syn_syn_easy_dataframe] = \
        create_dataframes_without_duplicates(zip([syn_syn], [syn_syn_easy]),
                                             easy_hard_split,
                                             'label1',
                                             'label2')

    [fsn_syn_dataframe], [fsn_syn_easy_dataframe] = \
        create_dataframes_without_duplicates(zip([fsn_syn], [fsn_syn_easy]),
                                             easy_hard_split,
                                             'pref',
                                             'alt')

    save_positive_instances(dataset_path,
                            easy_hard_split,
                            split_distance,
                            [syn_syn_dataframe, fsn_syn_dataframe],
                            [syn_syn_easy_dataframe, fsn_syn_easy_dataframe],
                            ['SYN_SYN', 'FSN_SYN'])
