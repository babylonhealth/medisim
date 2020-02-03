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

"""Create a collection of binary medical term similarity datasets."""

import os
import sys
import argparse

sys.path.append('..')

from dataset_creation_from_SNOMED.positive_instances_from_labels import positive_instances_from_labels
from dataset_creation_from_SNOMED.positive_instances_from_substitutions import positive_instances_from_substitutions
from dataset_creation_from_SNOMED.negative_sampling_from_positive_instances import negative_instances


parser = argparse.ArgumentParser(description='Similarity dataset creation from SNOMED')

parser.add_argument("--snomed_path", type=str, default="../SNOMED_files/",
                    help="Path to input folder containing SNOMED files")
parser.add_argument("--dataset_path", type=str, default="SNOMED_datasets/",
                    help="Path to output folder for new datasets")

# Changing these arguments results in a different dataset!
parser.add_argument("--easy_hard_split", type=bool, default=True,
                    help="Split into easy/hard datasets")
parser.add_argument("--split_distance", type=int, default=5,
                    help="Max Levenshtein distance for easy instances")
parser.add_argument("--neg_sampling_strategies", type=list, default=['advanced', 'simple'],
                    help="Strategies to use for negative sampling")

params = parser.parse_args()

if not os.path.isdir(params.dataset_path):
    os.mkdir(params.dataset_path)

print('*** Starting creation of positive instances from concept labels ***\n')
positive_instances_from_labels(easy_hard_split=params.easy_hard_split,
                               split_distance=params.split_distance,
                               snomed_path=params.snomed_path,
                               dataset_path=params.dataset_path)

print('*** Starting creation of positive instances from concept substitutions ***\n')
positive_instances_from_substitutions(easy_hard_split=params.easy_hard_split,
                                      split_distance=params.split_distance,
                                      snomed_path=params.snomed_path,
                                      dataset_path=params.dataset_path)

print('*** Starting creation of negative instances ***\n')
negative_instances(dataset_path=params.dataset_path,
                   strategies=params.neg_sampling_strategies)
