Tutorial
========

Installation
------------

CPU Only
~~~~~~~~

Install `Docker <https://docs.docker.com/get-docker/>`__ (at least version 19.03).

GPU
~~~

Make sure that NVIDIA driver (version >= 450) is installed.
After that install `Docker <https://docs.docker.com/get-docker/>`__ (at least version 19.03) and `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`__.
Ensure that ``docker-compose`` version is at least 1.28.
See `the documentation <https://docs.docker.com/compose/install/>`__ for instructions on how to upgrade it.

.. _paramspec:

Parameters Specification
------------------------

Database Password
~~~~~~~~~~~~~~~~~

Please select a password and write it to ``postgres_db/postgres_password.txt``.

Data Folder Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

Set environment variable ``SHIFT_MOUNT_LOCATION`` to the **absolute** path of a folder that contains the input data (CSV file, images).
Requests via API should provide a path relative to the ``SHIFT_MOUNT_LOCATION``. In case that you do not intend to use your own data, please create an empty folder and set the variable to its path.

.. _GPUEnviron:

GPU Devices (GPU Only)
~~~~~~~~~~~~~~~~~~~~~~

Set ``SHIFT_DEVICES`` environment variable like you would usually set ``CUDA_VISIBLE_DEVICES`` to specify which devices should be visible to sh¡ft!. If only CPU should be used, simply set it to ``""``.

.. _parallel:

Parallelism Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

Set environment variable ``SHIFT_PARALLELISM`` to a number that is at least as large as the value of ``MAX_CPU_JOBS`` in the ``.env`` file and the number of GPU devices specified via the ``SHIFT_DEVICES`` environment variable from :ref:`gpuenviron`.

Custom Location for Results (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To specify a custom location for results, please replace the last part of the ``docker-compose.yml`` file with

.. code::

    volumes:
        results:
            driver: local
            driver_opts:
                type: bind
                o: bind
                device: <path to results location>
        cache:
            driver: local
            driver_opts:
                type: bind
                o: bind
                device: <path to cache location>
        postgres:
        flower:
        redis:

as explained in `this StackOverflow answer <https://stackoverflow.com/questions/38396139/docker-change-folder-where-to-store-docker-volumes/38396602#38396602>`__.
The folder that is referenced by the path must be empty and created beforehand (automatic creation is not possible).
Please note that the following option is only available on Linux (see `here <https://docs.docker.com/engine/reference/commandline/volume_create/#driver-specific-options>`__) and that removing the volume with ``docker volume rm shift_results`` will not delete the created files within folders.

Build
-----

.. code::

    docker-compose build

to build all images.

Run
---

.. caution::
    The system is not meant to be exposed to the internet.
    Instead, if the system is run on a remote machine, one is supposed to use for example SSH local port forwarding.

.. danger::
    Before using the system, please also make sure to understand what `influence Docker has on firewall <https://github.com/moby/moby/issues/22054>`__.

CPU Only
~~~~~~~~

Run

.. code::

    docker-compose up

to start the containers. Press ``Ctrl + C`` once to stop them.

GPU
~~~

Replace the previous command with

.. code::

    docker-compose -f docker-compose.yml -f gpu.yml up

Example Request
~~~~~~~~~~~~~~~

.. caution::
    Before running example below, make sure to set the right batch size with ``/register_text_model/``.

.. code::

    {
        "train": [
            {
                "reader": {
                    "tf_dataset_name": "glue/sst2:1.0.0",
                    "split": "train",
                    "embed_feature_path": ["sentence"],
                    "label_feature_path": ["label"]
                }
            }
        ],
        "test": [
            {
                "reader": {
                    "tf_dataset_name": "glue/sst2:1.0.0",
                    "split": "validation",
                    "embed_feature_path": ["sentence"],
                    "label_feature_path": ["label"]
                }
            }
        ],
        "models": [{"tf_text_name": "NNLM 50"}],
        "classifiers": ["Euclidean NN", "Cosine NN"]
    }

This JSON will trigger inference on train and validation set of ``sst2`` dataset using ``NNLM 50`` model, and classifier jobs on the obtained data for both the Euclidean and cosine distance.


Monitoring
----------

While running, visit `localhost:8001 <http://localhost:8001>`__ to monitor the status of jobs and workers via `Flower <https://github.com/mher/flower>`__.

API Documentation
-----------------

If containers are running, API documentation can be found at `localhost:8000/redoc <localhost:8000/redoc>`__ and `localhost:8000/docs <http://localhost:8000/docs>`__.
