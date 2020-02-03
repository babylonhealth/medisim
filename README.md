# medisim :pill:
**medisim** is a collection of new large-scale **medi**cal term **sim**ilarity dataset based on SNOMED-CT.
The code in this repository creates the collection of datasets.

If you use one of these datasets in your research, please cite

```
@inproceedings{
SchulzJuric2020AAAI,
title={Can Embeddings Adequately Represent Medical Terminology? New Large-Scale Medical Term Similarity Datasets Have the Answer!},
author={Claudia Schulz and Damir Juric},
booktitle={34th AAAI Conference on Artificial Intelligence},
year={2020},
pages={to appear}
}
```

**Contact:** Claudia Schulz <claudia.schulz@babylonhealth.com> or Damir Juric <damir.juric@babylonhealth.com>


## Dependencies

This code is written in Python 3.7. The requirements are listed in `requirements.txt`.
```
pip3 install -r requirements.txt
```

Download the SNOMED-CT version 20190131 (https://utslogin.nlm.nih.gov/cas/login) and place the following files in
the `/SNOMED_files` directory:
* sct2_Concept_Full_INT_20190131.txt
* sct2_Description_Full-en_INT_20190131.txt
* der2_cRefset_AttributeValueFull_INT_20190131.txt

**Note:** if you use a different SNOMED-CT version, this will result in *different* datasets!

## Creating the Term Similarity Datasets
The `/dataset_creation_from_SNOMED` directory contains all code and data required to construct binary term similarity
 datasets from SNOMED CT.
All datasets are tab-separated, containing instances of the form 

`term1   term2   1` (positive instance) 
or `term1   term2   0` (negative instance).

The dataset creation is done in two steps: 
1) creation of **positive instances** from SNOMED concept labels and from
 SNOMED concept deletions and
 2) creation of **negative instances** from the positive ones using *simple* random sampling
 and/or *advanced* sampling based on Levenshtein distance.

To create all datasets run:
```
python3 create_datasets.py --snomed_path [location of SNOMED files if not ../SNOMED_files/] --dataset_path [location to save new datasets]
```

There are further arguments to control the dataset creation, however changing these will result in *different* datasets!

### Detail on Positive Instances
`positive_instances_from_labels.py` and `positive_instances_from_deletions.py` create term pairs that
 form the positive instances in the datasets.


#### Positive Instances from Labels
`positive_instances_from_labels()` creates two datasets:
* **FSN-SYN**: for each active SNOMED concept *c*, create pairs *FSN(c) - synonym(c)* using the one fully specified name (FSN)
 and all the synonyms of *c*
* **SYN-SYN**: or each active SNOMED concept *c*, create pairs *label1(c) - label2(c)* using all labels (i.e. synoyms and FSN)
of *c*

The `sct2_Concept_Full_INT_20190131.txt` file is used to check if concepts are active and that they are
in the *core module* rather than the *model component module*, which consists of properties
 and descriptional concepts like 'Inactive Value'.
Concept labels are then extracted from the `sct2_Description_Full-en_INT_20190131.txt` file.

In most cases FSNs end with a parenthesis indicating the concept's semantic type and there is exactly the same
 synonym without the paranthesis.
Thus, parenthesese are deleted from the FSN and the same synonym is disregarded to not create pairs of
 exactly the same labels.


#### Positive Instances from Deleted Concepts and their Replacement
`positive_instances_from_substitutions()` creates three datasets from concepts that got deleted in 
SNOMED and are replaced by another concept, resulting in similar concept pairs of the form
*deletedConceptFSN - replacementConceptFSN*. SNOMED specifies various reasons for deletion and replacement.
 We extract concept pairs of the following reasons: 
* **same_as**: a concept is deleted as it is a *duplicate*, the concept is specified to to be the *same as* its replacement
* **possibly_equivalent_to**: a concept is deleted as it is *ambiguous*, the concept is specified to be *possibly equivalent to* 
its replacement
* **replaced_by**: a concept is deleted as it is *outdated* or *erroneous*, the concept is specified to be *replaced by* 
its replacement 

The type of replacement and the replacing concept are specified in `der2_cRefset_AssociationFull_INT_20190131.txt`. 

For each deleted concept and its replacement concept, the FSNs of both concepts are used, which are obtained from
the `sct2_Description_Full-en_INT_20190131.txt` file.
Again, parenthesis indicating the concept's semantic type are deleted from the FSN, as well as '[D]' in the label, which 
indicates that the concept is deprecated. 


### Detail on Negative Instances
`negative_sampling_from_positive_instances()` creates negative instances from the
positive ones using two strategies:

1) **random** sampling: the first term `term1` of each positive term pair is randomly matched with another
 term `termX` that does not form a positive instance with `term1`.\
 This creates datasets with names ending `_simple`.
2) **Levenshtein** sampling: the first term `term1` of each positive term pair is matched with a term `termX` that has 
smallest Levenshtein distance to `term1`, while not forming a positive instance with `term1` or with any of its 
similar terms (as given by the positive instances containing `term1`).\
 This creates datasets with names ending `_advanced`.
