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

"""Definition of SNOMED code meanings"""

from enum import Enum

class SnomedID(Enum):
    MODEL_COMPONENT_MODULE = 900000000000012004
    SYNONYM_DESCRIPTION = 900000000000013009
    FSN_DESCRIPTION = 900000000000003001
    POSSIBLY_EQUIVALENT_TO_REFSET = 900000000000523009
    SAME_AS_REFSET = 900000000000527005
    REPLACED_BY_REFSET = 900000000000526001
