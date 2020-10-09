## Olympus
[![Build Status](https://travis-ci.com/FlorianHase/olympus.svg?token=bMWWqBdm3xytautMLsPK&branch=dev)](https://travis-ci.com/FlorianHase/olympus)
[![codecov](https://codecov.io/gh/FlorianHase/olympus/branch/flo/graph/badge.svg?token=FyvePgBDQ5)](https://codecov.io/gh/FlorianHase/olympus)

Docs: https://florianhase.github.io/olympus/

## Emulator status

Status options:
* `pending`:    dataset has been identified
* `parsed`:     dataset is in a machine readable format
* `integrated`: dataset is already available in olympus
* `trained`:    an emulator has been trained to acceptable accuracy

|    | Dataset              | Name         | Model | Objective       | Status     | Test R2 score |
|----|----------------------|--------------|-------|-----------------|------------|---------------|
|[ 1]| hplc_n9              | hplc         | bnn   | peak area       | trained    | 0.98          |
|[ 2]| alkox                | ??           | bnn   | reaction rate   | trained    | 0.94          |
|[ 3]| snar                 | snar         | bnn   | e_factor        | trained    | 0.99          |
|[ 4]| color mixing (bob)   | colormix_bob | bnn   | green-ness      | trained    | 0.94          |
|[ 5]| color mixing (n9)    | colormix_n9  | bnn   | green-ness      | trained    | 0.94          |
|[ 6]| demello set          | ??           | bnn   | yield of X3     | trained    | 0.99          |
|[ 7]| n_benzylation        | ??           | bnn   | e_factor        | trained    | 0.98          |
|[ 8]| suzuki               | suzuki       | bnn   | %AY             | trained    | 0.99          |
|[ 9]| excitonics           | excitonics   | nn    | efficiency      | trained?   | 0.69          |
|[10]| ptc                  | ??           | bnn   | TBD             | trained    | 0.99          |
|[11]| photobleaching pce10 | photobl      | bnn   | stability       | trained    | 0.93          |
|[12]| photobleaching wf3   | photobl      | bnn   | stability       | trained    | 0.89          |
|[13]| ada thin film?       | ??           | bnn   | pseudo-mobility | trained    | 0.64          |

## Planners

Status options:
* `pending`: not implemented yet
* `tested`:  implemented and unit test added
* `debugging`: implemented but not passing tests/problematic

|    | Name             | Status     |
|----|------------------|------------|
|[ 1]| Basin hopping    | tested     |
|[ 2]| Conj. gradient   | tested     |
|[ 3]| CMA-ES           | tested     |
|[ 4]| Diff. evolution  | tested     |
|[ 5]| GPyOpt           | tested     |
|[ 6]| Hyperopt         | tested     |
|[ 7]| Latin hypercube  | tested     |
|[ 8]| LBFGS            | tested     |
|[ 9]| Phoenics         | tested     |
|[10]| PSO              | tested     |
|[11]| Deap             | tested     |
|[12]| Random sampling  | tested     |
|[13]| Simplex          | tested     |
|[14]| SLSQP            | tested     |
|[15]| SMAC             | debugging  |
|[16]| Snobfit          | tested     |
|[17]| Sobol sampling   | tested     |
|[18]| Steepest descent | tested     |
