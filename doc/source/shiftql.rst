SHIFT-QL
========

   The Query Language for Transfer Learning

Specifications
--------------

.. code::

   USE {hostname};
   SELECT {type} FROM {table_name} WHERE {key=value} AND {key2=value2} ORDER BY {func}({metric}) ON {task_type} {dataset_name};

The possible values are listed:

+-------+------------------------+------------------------------------+
| Var   | Description            | Examples (possible values)         |
| iable |                        |                                    |
| Name  |                        |                                    |
+=======+========================+====================================+
| hos   | The address of the     | http://127.0.0.1                   |
| tname | server                 |                                    |
+-------+------------------------+------------------------------------+
| type  | which columns to query | ALL (default), ALL+VEC (append the |
|       | from                   | vector representation)             |
+-------+------------------------+------------------------------------+
| table | Where to query         | image_models; text_models;         |
| _name |                        |                                    |
+-------+------------------------+------------------------------------+
| key=  | query conditions       | source=“torchvision”               |
| value |                        |                                    |
+-------+------------------------+------------------------------------+
| func  | Function               | AVG, MAX, etc.                     |
+-------+------------------------+------------------------------------+
| m     | accuracy,              |                                    |
| etric | linear_accuracy,       |                                    |
|       | knn_accuracy           |                                    |
+-------+------------------------+------------------------------------+
| task  | task-agnostic or       | benchmark (task-agnostic) or task  |
| _type | task-aware             | (task-aware)                       |
+-------+------------------------+------------------------------------+
| da    |                        | imagenet, etc.                     |
| taset |                        |                                    |
| _name |                        |                                    |
+-------+------------------------+------------------------------------+

Restrict model pool
~~~~~~~~~~~~~~~~~~~

Examples:

-  Only PyTorch models;
-  Models w/ less than 1M parameters
-  Models published before 1.1.2021 *and* after 1.1.2018

.. code::

   SELECT ALL FROM image_models WHERE source="torchvision" AND parameters<=100000 AND xxx;

A: Task-Agnostic Search
~~~~~~~~~~~~~~~~~~~~~~~

Examples:

-  Top 2 highest ImageNet accuracies
-  Most robust and best ImagetNet model
-  Largest model
-  Latest model *or* top 1 highest ImageNet accuracies (possibly
   disjoint)

.. code::

   SELECT ALL FROM image_models ORDER_BY number_parameters DESC;

.. code::

   SELECT ALL FROM image_models ORDER_BY accuracy ON ImageNet DESC;

(new keyword: ON)

B: Meta-Learned Task-Agnostic Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples:

-  Top 1 average ‘structured’ (benchmark) dataset

-  Top 1 min all datasets

-  Top 2 max ‘natural’ dataset *or* average top 1 all dataset (possibly
   disjoint)

.. code::

   DECLARE datasets AS SELECT ALL FROM datasets WHERE source=“natural”;
   SELECT ALL FROM image_models ORDER BY MAX(accuracy) ON BENCHMARK datasets;
   SELECT ALL FROM image_models ORDER BY MEAN(accuracy) ON BENCHMARK datasets; 

C: Meta-Learned Task-Aware Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples:

-  Top 1 most similar (benchmark) dataset

-  Top 1 second most similar ‘natural’ dataset

-  Top 2 most similar dataset *without* top 1 most similar ‘natural’
   dataset

.. code::

   DECLARE natural_datasets AS SELECT ALL+VEC FROM datasets WHERE type="natural";
   DECLARE dataset AS SELECT ALL+VEC FROM datasets ORDER BY DIST(VEC) LIMIT 1;
   SELECT from image_models ORDER BY accuracy on TASK dataset;

D: Task-Aware Search
~~~~~~~~~~~~~~~~~~~~

Examples:

-  Top 2 linear accuray

-  Top 1 kNN accuracy (Snoopy)

-  Top 2 linear accracy *or* top 1 kNN accuracy (disjoint?)

.. code::

   SELECT from image_models ORDER BY linear_accuracy on TASK dataset;