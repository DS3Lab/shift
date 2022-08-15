# Which SHiFT-QL Query to Use

This guideline for choosing which search strategy (SHiFT-QL query) to use in SHiFT is compiled based on our own experience, as well as best practices for transfer learning available in online blogs (e.g., [1,2,3]) and large scale transfer learning studies [4].

## General

In general, the transfer learning literature suggests to transfer from a model trained on a *similar* dataset. Similarity is rather wage. Obviously, if one happens to have access to a high accuracy pre-trained model on the same dataset, this one can be used without any fine-tuning to yield high accuracy predictions. If no such model exists, best practices and surveys suggest using a model pre-trained on the smallest distribution in which the target distribution is included. To give an example: let's assume the target dataset contains flowers. It is better to use a model pre-trained on any plants compared to one trained on all living things (e.g., animals and plants).

## The Challenge

The challenge with pre-trained models hubs and SHiFT, is that the upstream dataset and distributions are usually not available for users to compare to their downstream dataset. Furthermore, there might be no upstream distribution in which the downstream one is included. To still enable successful search over the models, we provide some simple suggestions to follow.

## Guidelines

We next provide some guidelines pointing to the SHiFT-QL queries numbered in the SHiFT paper, which could in many cases yield to a good tradeoff between computational requirement to run the search query and accuracy of the model returned.

### No knowledge

Let us start with the simplest case: If a user of SHiFT has no knowledge about the model pool or relation between downstream dataset and the benchmark datasets, it is the safest to run the hybrid search query (Q4) over all suitable models and setting one trained on the largest corpus as the task-agnostic one. If no model fits as a task-agnostic one, users should use a linear proxy model (Q3), or a cheaper alternative (Q2).

### Homogeneous model pool

If all models filtered by the user in a model pool were trained on the same upstream dataset, users should usually can retrieve the best model using a simple task-agnostic search query by ranking based on upstream accuracy (Q1). If on the other hand, all models have the same architecture and users have access to the number of upstream samples, this can, again in a task-agnostic manner, be used to rank models accordingly.

### Similar benchmark dataset(s)

If there exists a similar benchmark datasets in SHiFT (e.g., PETS), the ranking of that model can be used as is to retrieve the best model(s). If the user assumes a very similar dataset exists, but does not know which one, techniques like Task2Vec can help retrieve this similar dataset (Q5).
Finally, if a user has a structured dataset and knows that there are multiple of them in the set of benchmark datasets, it might be worth filtering the best models for each of the structured dataset and fine-tune all of them (Q8), or alternatively, if this is computationally too demanding, run a linear proxy over those (Q7).

## References:

- [1] https://towardsdatascience.com/how-to-choose-the-best-keras-pre-trained-model-for-image-classification-b850ca4428d4
- [2] https://www.dominodatalab.com/blog/guide-to-transfer-learning-for-deep-learning
- [3] https://towardsdatascience.com/how-to-apply-transfer-learning-successfully-6b6f489d287d
- [4] Mensink, T., Uijlings, J., Kuznetsova, A., Gygli, M., & Ferrari, V. (2021). Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types. IEEE Transactions on Pattern Analysis & Machine Intelligence, (01), 1-1.
