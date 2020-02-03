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

"""Creation of similar term pairs from SNOMED substituted concepts"""

import csv
import os
import glob
import pandas as pd
from tqdm import tqdm

from dataset_creation_from_SNOMED.snomed_id import SnomedID
from dataset_creation_from_SNOMED.positive_instances_utils import create_term_pairs
from dataset_creation_from_SNOMED.positive_instances_utils import save_positive_instances
from dataset_creation_from_SNOMED.positive_instances_utils import create_dataframes_without_duplicates
from dataset_creation_from_SNOMED.positive_instances_utils import clean_pref_term


def get_pref_label(concept, label_table):
    concept_labels = label_table[label_table["conceptId"] == concept]

    # sometimes the label is changed throughout the years, i.e. there are multiple entries for
    # a concept with multiple different labels
    # in this case choose the most recent one that is active and a pref label
    concept_pref_labels = []

    for label_id in concept_labels.id.unique():

        # get all entries of a label ID
        label_id_entries = concept_labels[concept_labels["id"] == label_id]

        # row index where this label was last updated (has the max effective time)
        last_update_entry = label_id_entries['effectiveTime'].idxmax()

        # if most recent entry of label is active and states it is the pref label, use this label
        if label_id_entries.loc[last_update_entry, 'active'] and \
                label_id_entries.loc[last_update_entry, 'typeId'] == SnomedID.FSN_DESCRIPTION.value:
            concept_pref_labels.append(label_id_entries.loc[last_update_entry].term)

    # make sure there is exactly one pref label
    if len(concept_pref_labels) == 0:
        raise Exception("No pref label found for concept: %s" %(concept))
    if len(concept_pref_labels) > 1:
        raise Exception("Multiple pref labels found for concept: %s" %(concept_pref_labels))

    return concept_pref_labels[0]


def clean_term(label):
    label_cleaned = clean_pref_term(label)
    # some labels are tagged with [D] as deprecated and mostly they are replaced by
    # non-deprecated concepts with the same label, so ignore these instances
    if label_cleaned.startswith("[D]"):
        label_cleaned = label_cleaned.split("[D]", 1)[1].strip()
    if label_cleaned.endswith("[D]"):
        label_cleaned = label_cleaned.rsplit("[D]", 1)[0].strip()

    return label_cleaned


def read_syn_syn_instances(path):
    syn_syn_datasets = []
    for f in glob.glob(os.path.join(path, 'SYN_SYN*')):
        df = pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE,
                         keep_default_na=False, header=0,
                         names=['source', 'target'])
        syn_syn_datasets.append(df)
    return pd.concat(syn_syn_datasets, axis=0, ignore_index=True)


def is_pair_in_syn_syn(syn_syn_instances, source_label_text, target_label_text):
    return is_same_pair(syn_syn_instances, source_label_text, target_label_text) or \
           is_same_pair(syn_syn_instances, target_label_text, source_label_text)

def is_same_pair(syn_syn_instances, label1, label2):
    return not syn_syn_instances[(syn_syn_instances['source'] == label1)
                                 & (syn_syn_instances['target'] == label2)].empty


def is_active(substitutes_core_module, source_id, target_id, deletion_reason):

    substitution_pair_rows = \
        substitutes_core_module[(substitutes_core_module['referencedComponentId'] == source_id)
                                & (substitutes_core_module['targetComponentId'] == target_id)
                                & (substitutes_core_module['refsetId'] == deletion_reason)]

    return substitution_pair_rows.loc[substitution_pair_rows["effectiveTime"].idxmax()].active



##################################################################
# MAIN
##################################################################


def positive_instances_from_substitutions(easy_hard_split,
                                          split_distance,
                                          snomed_path,
                                          dataset_path):
    # input SNOMED files
    labels = \
        pd.read_csv(os.path.join(snomed_path, "sct2_Description_Full-en_INT_20190131.txt"),
                    sep="\t", header=0,
                    quoting=csv.QUOTE_NONE, keep_default_na=False)
    substitutes = \
        pd.read_csv(os.path.join(snomed_path, "der2_cRefset_AssociationFull_INT_20190131.txt"),
                    sep="\t", header=0,
                    quoting=csv.QUOTE_NONE, keep_default_na=False)

    # get already created positive instances from labels to avoid duplicate term pairs
    syn_syn_instances = read_syn_syn_instances(dataset_path)

    # dictionaries to capture extracted label pairs
    possibly_equivalent_to = {'source':[], 'target':[]}
    same_as = {'source':[], 'target':[]}
    replaced_by = {'source':[], 'target':[]}
    possibly_equivalent_to_easy = {'source': [], 'target': []}
    same_as_easy = {'source': [], 'target': []}
    replaced_by_easy = {'source': [], 'target': []}

    # only use core module (rather than model componenent module)
    substitutes_core_module = substitutes[substitutes["moduleId"] !=
                                          SnomedID.MODEL_COMPONENT_MODULE.value]

    # get all pairs of source - target concept pairs
    # NOTE: this may drop a source-target instance that is active,
    # whereas the inactive one remains in substitution_pairs
    # therefore later we use substitute_concepts with max effectiveTime
    # to find an active association
    substitution_pairs = \
        substitutes_core_module.drop_duplicates(subset=['referencedComponentId', 'targetComponentId'])

    # go through all the associations and
    # select the relevant ones between replaced concept pairs
    # with the desired reasons
    for index, substitution_pair in tqdm(substitution_pairs.iterrows(),
                                         total=substitution_pairs.shape[0]):

        # check if a source - target pair has one of the desired deletion reasons
        deletion_reason = substitution_pair.loc['refsetId']
        if deletion_reason == SnomedID.POSSIBLY_EQUIVALENT_TO_REFSET.value:
            add_to = possibly_equivalent_to
            add_to_easy = possibly_equivalent_to_easy
        elif deletion_reason == SnomedID.SAME_AS_REFSET.value:
            add_to = same_as
            add_to_easy = same_as_easy
        elif deletion_reason == SnomedID.REPLACED_BY_REFSET.value:
            add_to = replaced_by
            add_to_easy = replaced_by_easy
        else:
            continue

        # get the IDs of the source and target concept
        source_id = substitution_pair.loc['referencedComponentId']
        target_id = substitution_pair.loc['targetComponentId']

        # only use source - target pairs whose most recent association is active
        if not is_active(substitutes_core_module, source_id, target_id, deletion_reason):
            continue

        # get pref label of source and target concept
        source_label = get_pref_label(source_id, labels)
        target_label = get_pref_label(target_id, labels)

        # the SNOMED substitution file contains some strange entries of e.g.
        # possEquivTo where the target is a namespace concept, these should be ignored
        if "namespace" in target_label:
            continue

        # get cleaned up labels of source and target
        source_label_cleaned = clean_term(source_label)
        target_label_cleaned = clean_term(target_label)

        # if the source and target labels are the same, ignore them
        if source_label_cleaned.lower() == target_label_cleaned.lower():
            continue

        # check if the current concept pair (or its reverse)
        # is already in the dataset of synonym labels
        if is_pair_in_syn_syn(syn_syn_instances, source_label_cleaned, target_label_cleaned):
            continue

        __add_to, _add_to_easy = \
            create_term_pairs([(source_label_cleaned, target_label_cleaned)],
                              easy_hard_split, split_distance,
                              'source', 'target', add_to, add_to_easy)


    normal_datasets, easy_datasets = \
        create_dataframes_without_duplicates(zip([possibly_equivalent_to, same_as, replaced_by],
                                                 [possibly_equivalent_to_easy,
                                                  same_as_easy,
                                                  replaced_by_easy]),
                                             easy_hard_split, 'source', 'target')

    save_positive_instances(dataset_path,
                            easy_hard_split,
                            split_distance,
                            normal_datasets,
                            easy_datasets,
                            ['possibly_equivalent_to', 'same_as', 'replaced_by'])
