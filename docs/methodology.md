# Methodology

This repository implements a two-stage structural-connectivity inference pipeline for neuronal populations observed through binned spike activity.

## 1. Activity matching with a differentiable SNN

The first stage treats the recurrent synaptic connectivity of a leaky integrate-and-fire (LIF) spiking neural network as the quantity to learn. Target spike trains are generated separately and reduced to binary 4 ms occupancy. Population bursts are detected using a NEST-style criterion based on the fraction of unique neurons active within non-overlapping 50 ms windows.

For each detected burst, the training sample is cropped from the first bin in the uninterrupted run of activity through the bin of maximum population activity. The recurrent LIF network is then optimized to reproduce the target temporal activity. Because hard spike generation is non-differentiable, training uses a surrogate-gradient approximation during backpropagation while retaining discrete spikes in the forward dynamics.

The learned recurrent weight matrix from each burst is saved. Repeating this process across bursts produces a distribution of learned weights for every candidate directed neuron pair rather than a single point estimate.

## 2. Edge-level feature engineering and supervised classification

For a candidate directed edge `i -> j`, repeated learned weights are summarized with statistics such as mean, minimum, maximum, standard deviation, and median. These summaries become the feature vector for a gradient-boosted tree classifier.

Ground-truth structural connectivity is read from the network YAML files. The classifier predicts an edge-existence probability for each directed pair, producing a probability matrix that can be thresholded or evaluated as a ranking.

## 3. Evaluation

The analysis scripts support:

- ROC AUC and precision-recall AUC;
- threshold sweeps with TPR, FPR, precision, recall and related statistics;
- total, undirected and direction-sensitive evaluations;
- source-neuron stratification for excitatory and inhibitory edges;
- graph-level comparisons including inferred connection density and clustering;
- aggregation across independent network instances with uncertainty estimates.

Train/test splitting is performed at the network-instance level where applicable. This is important: edge rows from the same network are highly dependent, so randomly splitting individual edges would leak network-specific information across the train and test sets.

## Why the two-stage design?

A direct correlation between two spike trains can reflect common input, indirect pathways, or population-level dynamics rather than a direct synapse. The activity-matching SNN introduces a mechanistic intermediate representation: the weights required for a dynamical model to reproduce the observed activity. The supervised stage then asks whether patterns in those learned weights consistently distinguish true structural edges from non-edges.
