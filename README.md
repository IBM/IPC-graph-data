# IPC: A Graph Data Set Compiled from International Planning Competitions

This repository provides the data set, named IPC, for the evaluation of machine learning methods on graphs. It contains 2439 labeled graphs, presplit for training/validation/testing. The node counts have a highly skewed distribution, ranging from less than ten to a few hundred thousands. Each graph corresponds to one planning problem and it is associated with a set of 17 targets, each of which is the time for a certain planner to solve the problem. IPC serves as a benchmark data set for graph representation learning and other machine learning tasks.

## <a id="background"></a>Background

Automated planning is one of the foundational areas of AI. Since a single planner unlikely works well for all problems and domains, portfolio-based techniques become increasingly popular. In particular, deep learning emerges as a promising methodology for online planner selection. A prominent example is the winner, *Delfi* [(Katz et al. 2018)](#Katz2018), of the Optimal Track of the [International Planning Competition (IPC) 2018](https://ipc2018.bitbucket.io).

Planning systems achieve state-of-the-art performance by using structural graph representations of the planning problems [(Ma et al. 2020)](#Ma2020), [(Katz et al. 2018)](#Katz2018). Two examples are the *problem description graph* [(Pochter et al. 2011)](#Pochter2011) for a **grounded** representation, and the *abstract structure graph* [(Sievers et al. 2017)](#Sievers2017) for a **lifted** representation.

Here, we release these graphs constructed from the problems occurred in the competitions. In particular, the historical IPC problems form the training and validation sets, whereas those of the year 2018 form the test set. A small amount of problems are ignored, the reason of which is explained toward the end of this section. When performing the training/validation split, problems in the same domain are not separated in two sets. Hence, we randomly select a few domains to form the validation set, such that its size is approximately 10% of that of the training set.

The graphs have node labels, which are converted to one-hot node feature vectors needed by some graph neural network architectures.

There are 17 planners in the portfolio, each of which produces for a planning problem a target value, which is the time needed for solving the problem. In other words, each graph has 17 target values. The details of the planners may be found in [Katz et al. (2018)](#Katz2018).

The timeout limit for each problem is 1800s. For planners that fail to solve the problem before timeout, the target value is artificially set as 10000.

The problems not solved by any of the planners in the portfolio within the timeout limit 1800s are ignored in the construction of the data set. In particular, some of these problems occur in IPC 2018. Hence, the test set contains problems strictly fewer than those in IPC 2018.

## File Format

There are two folders `grounded` and `lifted`, each of the which is a complete data set, already split for training/validation/testing, as the names of the files inside the folder suggest. The two folders correspond to the same set of planning problems, differing only in the graph representation. See the section [Background](#background) for more details.

When loaded with python, for example:

```python
import json
import gzip
with gzip.open('ipc-grounded-valid.json.gz') as f:
    data = json.load(f)
```

`data` contains a list of graphs. For the i<sup>th</sup> graph,

- `data[i]['graph']` is a list of tuples `[source_node, edge_label, target_node]`;
- `data[i]['targets']` is a list of `[target_value]`;
- `data[i]['node_features']` is a list of `[node_feature_vector]`.

Note: This data set has no labels for edges. Hence, all `edge_label`s have a place-holder value 1.

Additionally, the folder `problems` contains files that list the domain/problem name of each graph, in plain text. Each row corresponds to one graph, with a space separating the domain name and the problem name.

## Example Tasks

This data set may be used for a variety of tasks, two examples given in the following.

**Example 1: Multioutput regression.** Build a regression model that takes a graph as input and predicts a vector of target values.

**Example 2: Multilabel classification.** This task is used for online planner selection [(Ma et al.2020)](#Ma2020), [(Katz et al. 2018)](#Katz2018), [(Sievers et al. 2019)](#Sievers2019): Given a planning task (graph), one selects from a portfolio of planners whose run time (target value) is no greater than the timeout limit (1800). For this task, one may construct a binary labelling vector from the vector of target values, measured on whether the target value is greater than the timeout limit, build a multioutput binary classification model, and select the class with the lowest predicted probability. Benchmark results may be found from [Ma et al. (2020)](#Ma2020). There are, of course, several straightforward variants for approaching this selection problem. For example, one may change from the multilabel formulation to the multiclass formulation.

## Citing This Data Set

```
@InProceedings{Ma2020,
  author    = {Tengfei Ma and Patrick Ferber and Siyu Huo and Jie Chen and Michael Katz},
  title     = {Online Planner Selection with Graph Neural Networks and Adaptive Scheduling},
  booktitle = {Proceedings of the Thirty-Fourth {AAAI} Conference on Artificial Intelligence},
  year      = {2020},
}

@InProceedings{Ferber2019,
  author    = {Patrick Ferber and Tengfei Ma and Siyu Huo and Jie Chen and Michael Katz},
  title     = {{IPC}: A Benchmark Data Set for Learning with Graph-Structured Data},
  booktitle = {ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Data},
  year      = {2019},
}

@InProceedings{Sievers2019,
  author    = {Silvan Sievers and Michael Katz and Shirin Sohrabi and Horst Samulowitz and Patrick Ferber},
  title     = {Deep Learning for Cost-Optimal Planning: Task-Dependent Planner Selection},
  booktitle = {Proceedings of the Thirty-Third {AAAI} Conference on Artificial Intelligence},
  year      = {2019},
}
```

## Bibliography

- <a name="Katz2018"></a>Michael Katz, Shirin Sohrabi, Horst Samulowitz, and Silvan Sievers. [Delfi: Online planner selection for cost-optimal planning](https://ipc2018-classical.bitbucket.io/planner-abstracts/teams_23_24.pdf). In Ninth International Planning Competition (IPC-9): planner abstracts, 2018.
- <a name="Ma2020"></a>Tengfei Ma, Patrick Ferber, Siyu Huo, Jie Chen, and Michael Katz. [Online Planner Selection with Graph Neural Networks and Adaptive Scheduling](https://arxiv.org/pdf/1811.00210.pdf). In AAAI 2020.
- <a name="Pochter2011"></a>Nir Pochter, Aviv Zohar, and Jeffrey S. Rosenschein. [Exploiting problem symmetries in state-based planners](http://icaps11.icaps-conference.org/proceedings/hdip/pochter-et-al.pdf). In AAAI, 2011.
- <a name="Sievers2017"></a>Silvan Sievers, Gabriele Röger, Martin Wehrle, and Michael Katz. [Structural symmetries of the lifted representation of classical planning tasks](http://ai.cs.unibas.ch/papers/sievers-et-al-icaps2017wshsdip-a.pdf). In ICAPS 2017 Workshop on Heuristics and Search for Domain-independent Planning, 2017.
- <a name="Sievers2019"></a>Silvan Sievers, Michael Katz, Shirin Sohrabi, Horst Samulowitz, and Patrick Ferber. [Deep Learning for Cost-Optimal Planning: Task-Dependent Planner Selection](). In AAAI, 2019.

## Contributors (In Alphabetical Order)

- Jie Chen, IBM Research
- Patrick Ferber, University of Basel
- Michael Katz, IBM Research
- Horst Samulowitz, IBM Research
- Silvan Sievers, University of Basel
- Shirin Sohrabi, IBM Research

## Contact

Please direct questions to Jie Chen, chenjie@us.ibm.com.
