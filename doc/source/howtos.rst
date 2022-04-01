HOWTOs
======

Using the System
----------------

How Do I Use ``tox``?
~~~~~~~~~~~~~~~~~~~~~

To check whether everything is correctly formatted, to run unit tests and to rebuild the documentation we use `tox <https://tox.readthedocs.io/en/latest/>`_.

To use it, create a Python **3.6** environment and install ``tox`` with

.. code::

    pip install tox


To run everything, simply run

.. code::

    tox

using the new environment while being positioned in the ``shift`` folder. ``tox`` will automatically create all required environments and install dependencies in the ``.tox`` folder.

To run specific steps use:

* ``tox -e flake8`` to run the `Flake8 <https://flake8.pycqa.org/en/latest/>`__ linter.
* ``tox -e black-check`` to check formatting with `Black <https://black.readthedocs.io/en/stable/>`__.
* ``tox -e isort-check`` to check whether imports could be optimized with `isort <https://pycqa.github.io/isort/>`__.
* ``tox -e <folder name>-mypy`` to check types with `mypy <http://mypy-lang.org/>`__.
* ``tox -e <folder name>-test`` to run unit tests.
* ``tox -e docs`` to generate documentation.

To see all steps that can be run, please refer to the ``tox.ini`` file.

To fix formatting problems use:

* ``tox -e black`` to fix formatting with `Black <https://black.readthedocs.io/en/stable/>`__.
* ``tox -e isort`` to optimize imports with `isort <https://pycqa.github.io/isort/>`__.

Using ``tox`` on Mac
####################

PyTorch uses different libraries on Mac than on Linux and Windows. Consequently, in file ``worker_general/requirements-dev.txt`` part

.. code::

    torch==1.7.1+cpu
    torchvision==0.8.2+cpu

has to be changed to

.. code::

    torch==1.7.1
    torchvision==0.8.2

What Data Can I Obtain From the Progress Endpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Progress endpoint always returns the status of the job and in some cases also additional information.
When the job finishes, the ``additional`` field contains:

* Hash which can be used to locate the embedded data in case of **inference requests/jobs**.
* Error of the classifier in case of the **classifier requests/jobs**.

It can also contain some additional information about status (e.g. amount of data processed) while the job is still running.

How Is Logging Used?
~~~~~~~~~~~~~~~~~~~~

By default, all containers use ``INFO`` level for showing logs.
Celery workers are configured to report ``stdout`` and ``stderr`` as ``DEBUG`` level logs.
TensorFlow is configured to only print logs signalling that something is or might be wrong.

In case of bugs, you can lower the level to ``DEBUG``, which will show additional logs as well as ``stdout`` and ``stderr``.

.. _NumPyFormat:

How Does the NumPy Format Work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to be able to use NumPy with :py:class:`~common.pipeline.pipeline.io.NumPyReader`, some simple rules need to be obeyed:

1. Folder should contain a file ``dataset.txt`` to denote that the dataset is valid.
2. Files with ``.npz`` file extension within the folder are taken into account, files with other extensions and/or in subfolders are not.
3. Those files are sorted alphabetically (i.e. with Python ``sorted`` method) when reading.
4. Each file should contain same variable names (keys).
5. Each variable name within a file should contain data with the same first dimension (same number of data points).
6. Data from different files should of same shape, so that it can be concatenated.
7. Keys for embed and label feature should be the values of variables ``READER_EMBED_FEATURE_NAME`` (currently set to ``"embed"``) and ``READER_LABEL_FEATURE_NAME`` (currently set to ``"label"``) from :ref:`schemas`, respectively.
8. Preprocessing is not supported, all ingested data should be already preprocessed.

.. note::
    Since all ``.npz`` files within a folder are taken into account, :py:class:`~common.pipeline.pipeline.io.NumPyWriter` by default removes all files from the folder before writing to it.
    This is to ensure that if some job failed in the past and in the meantime either of ``RESULT_MAX_ROWS`` or ``RESULT_MAX_VALUES`` (see :ref:`envvars`) was changed, old ``.npz`` files are not taken into account.
    This is to prevent case when either of the environment variable values was increased and now less ``.npz`` files are generated than before.

    **Example**: Before failure, 10 files were written, and in the new job, only 5 files are written. This happened because of the increased values of environment variables.
    5 files from the previous run would then also be read as a part of the dataset even though they are invalid.

.. attention::
    Because ``.npz`` files are sorted alphabetically, make sure that numbers are padded if you use numbers for filenames, otherwise for instance ``10.npz`` will be used before ``2.npz``. This only applies when one wants to contruct the dataset from scratch, :py:class:`~common.pipeline.pipeline.io.NumPyWriter` ensures that automatically.

.. _envvars:

Which Environment Variables Can I Tweak?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tha majority of environment variables are set via the ``.env`` file. Here we cover only environment variables that can be tweaked from the ``.env`` file. For environment variables that should always be set, please refer to :ref:`paramspec`.

.. note::
    Environment variables presented here must be set directly in the ``.env`` file and cannot be overridden by setting the environment variable.

* ``MAX_CPU_JOBS``: Controls the maximum number of jobs executed concurrently on a CPU.
* ``TF_PREFETCH_SIZE``: Prefetch size used by `TensorFlow datasets <https://www.tensorflow.org/guide/data_performance#prefetching>`__.
* ``PT_PREFETCH_FACTOR``:  Prefetch factor used by `PyTorch datasets <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__.
* ``PT_NUM_WORKERS``: Number of workers used by `PyTorch datasets <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__.
* ``RESULT_MAX_ROWS`` and ``RESULT_MAX_VALUES``: Maximal number of rows and values (any type) that can be stored within a single file of the NumPy format (see :ref:`NumPyFormat`). Those values are intended to limit amount of data present in memory at any time. When writing to a file, the minimum of both rules is taken into account. There are two values so that the number of rows can adapt to the type of data (larger number of rows for embedded data, smaller number of rows for raw images).

.. attention::
    If more rows are added to the :py:class:`~common.pipeline.pipeline.io.NumPyWriter` at once than the maximum number of allowed rows, an error will be raised.

.. caution::
    Values specified via the ``.env`` file are assumed to be valid and are not checked at runtime, so be careful when changing them.

What Are The Rules Regarding Data When Using Classifier?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two simple rules:

1. Labels (and its changes) should have NumPy shape ``(#labels,)`` and NumPy type ``int64``.
2. Embedded data (and its changes) should have NumPy shape ``(#points, dimension)`` and NumPy type ``float32``.

There are additional rules when applying changes:

1. Specified change indices should be valid for the base underlying data - they should not be too large.
2. There should be the same number of change indices or less than there are data points specified by the change. This way each index has a corresponding data point.
3. Changes should have the same dimensions as the underlying data.

.. note::
    In case of label-only updates for the nearest neighbors classifier, the first rule is not checked. Invalid indices are in this case simply ignored.

.. tip::
    There can be less change indices than there are data points specified by the change. The same data (reader) can be then reused multiple times, each time with different change indices. Because of that, inference can be run only once.




What Are The Rules for Classifier RAM and GPU Memory Usage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nearest Neighbors
#################

In order for nearest neighbors to run, all test readers (``MutableReader``) and each train reader (``MutableReader``) separately must fit into RAM together.
When running the nearest neighbors algorithm, each train reader will be internally split so that each calculation will fit into the GPU memory.
In extreme case, train reader will be split into individual points, so there should be enough GPU memory to compute the nearest neighbors between all test points and a single training point.
If there is not enough GPU memory for such calculation, the classifier job will fail.

How Is Encoded Text Decoded?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Text encoding cannot be selected, it is always assumed that encoded texts are encoded using ``UTF-8``.

Extending Functionality
-----------------------

How Are GPUs Assigned To Different Tasks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in :ref:`gpuenviron`, environment variable ``SHIFT_DEVICES`` should be used in the same way as one would usually use ``CUDA_VISIBLE_DEVICES``.

Internally, ``gpu_ids`` property within ``docker-compose`` is set to ``SHIFT_DEVICES``.
Within containers, GPU devices are then enumerated starting from ``0``.
This means, for instance, that if ``SHIFT_DEVICES`` is set to ``3,5,6``, Docker containers with GPU enabled will see devices ``0``, ``1`` and ``2`` instead.
The change is captured within :py:class:`~scheduler.scheduler.DeviceManager`.

Each task receives together with the task arguments also a device ID from the scheduler, which uses the :py:class:`~scheduler.scheduler.DeviceManager`.
The GPU is then simply selected within the task by setting the environment variable ``CUDA_VISIBLE_DEVICES`` to the received device ID.

How Do Parallel Jobs Work?
~~~~~~~~~~~~~~~~~~~~~~~~~~

For each container there are ``SHIFT_PARALLELISM`` (see :ref:`parallel`) processes prepared by specifying the `-c flag <https://docs.celeryproject.org/en/stable/reference/cli.html#cmdoption-celery-worker-c>`__.
``SHIFT_PARALLELISM`` processes are needed per container, because at some point all jobs might be sent to just one container.

Scheduler takes care of scheduling jobs.
As soon as an appropriate device is free (either a GPU device or a CPU slot), the task gets sent to a queue that corresponds to the container (there is one queue per container) and one of the ``SHIFT_PARALLELISM`` processes starts to process it.

Why Are There ``py.typed`` Files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because of `Mypy <https://mypy.readthedocs.io/en/stable/installed_packages.html#making-pep-561-compatible-packages>`__.
This also explains why ``zip_safe = False`` is used.


Why Does ``Dockerfile`` Contain ``chmod`` Command?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When code is downloaded from GitHub as a ``.zip`` file, permissions are incorrect. According to `this GitHub issue <https://github.com/moby/moby/issues/6333>`__, permissions within container are inherited from the host.

Line

.. code::

    RUN chmod -R 644 . && chmod -R +X .

recursively sets permission ``644`` for files and ``755`` for folders. Since the user is changed after this line, at runtime files have permission ``4`` (read) and folders have permission ``5`` (read and execute).

Why Is Daemon Set To False At the Beginning of Celery Tasks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Line

.. code:: Python

    current_process().daemon = False

ensures that subprocesses spawned by Celery are not daemons. This way each task can use multiprocessing. This is needed for PyTorch datasets. More information `here <https://github.com/celery/celery/issues/1709#issuecomment-324802431>`__.

Why Are All Celery Tasks Within a ``try``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is to prevent `Celery failing because of custom exceptions <https://github.com/celery/kombu/issues/573>`__.
The task is then failed by raising ``RuntimeError`` which is serializable.

How Is the Startup Order of Containers Ensured?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the `Docker documentation <https://docs.docker.com/compose/startup-order/>`__, a combination of the ``depends_on`` and tool `dockerize <https://github.com/jwilder/dockerize>`__ is used to ensure that dependencies between containers are respected.

Scheduler and Rest containers depend both on PostgreSQL and Redis, so the ``dockerize`` tool is used.

Worker containers depend only on Redis database, however Celery only causes problems if Redis is not online at the time when the task is scheduled. Consequently, ``dockerize`` tool is not used, since the job is scheduled by the scheduler container, which does not run until Redis is online.

How Do I Write Unit Tests?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each package contains its own ``tests`` folder. Within the folder, each module within the package contains its own file ``test_<module name>.py`` that contains all unit tests for that module.


Environment variables are accessed via classes that override the `BaseSettings <https://pydantic-docs.helpmanual.io/usage/settings/>`__ class.
By default, environment variables set within the ``.env`` file are also used for testing, which can be seen in ``tox.ini`` file with lines

.. code::

    setenv = file|.env

Those values can be overridden by either patching the settings object or setting the variable in the ``tox.ini`` file as

.. code::

    ...
    setenv =
        file|.env
        SOME_VARIABLE=<some_value>
    ...

How Do I Add Custom Preprocessing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessing is defined by implementing the :py:class:`~common.pipeline.pipeline.model.PreprocessingSpecs` interface.
In sh¡ft!, preprocessing specifications for existing models are defined in the :py:mod:`general worker preprocessing module <worker_general.general.model.preprocessing>`, as preprocessing is only done in the general worker (TF 1.15 worker accepts only already preprocessed data).

The defined preprocessing specification should then be returned by a model via :py:meth:`~common.pipeline.pipeline.model.Model.get_preprocessing_specs`.

At the moment, there is only one preprocessing interface, which is used both for textual and image models.
In future, two separate interfaces could be used.

.. tip::
    It is a good idea to parametrize the preprocessing specifications. This way if two models require very similar preprocessing (e.g. resizing images to different sizes), the same class can be reused.

    Parameters passed to preprocessing can than be defined via the model configuration.

Textual Preprocessing
#####################

TensorFlow preprocessing should accept and return a TensorFlow tensor.
It is used with TFDS (and CSV) readers.
PyTorch preprocessing should accept and return a NumPy array/tensor.
It is used with HuggingFace readers.

.. important::
    The convention is that value ``None`` for a preprocessing function means no preprocessing.
    However, this might not be always suitable, as explained later.

.. caution::
    At the moment, text preprocessing is either a no-op operation or tokenization for one of the HuggingFace models.
    When defining preprocessing that returns text, pay attention to the types of NumPy/TensorFlow tensors.
    TensorFlow (TFDS and CSV) uses encoded strings with `NumPy dtype object_ <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.object_>`__, whereas HuggingFace uses unencoded strings with `NumPy dtype str_/unicode_ <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.str_>`__.

Image Preprocessing
###################

TensorFlow preprocessing should accept and return a TensorFlow tensor.
It is used with TFDS readers.
PyTorch preprocessing should accept a PIL image and return a PyTorch tensor.
It is used with torchvision and data folder readers.

.. attention::
    All preprocessing functions should return images with the following order of dimensions: ``(height, width, #channels)``.
    Models that accept images with different shape (e.g. torchivision models) are responsible for switching the dimensions.

.. attention::
    At the moment, all image models use some kind of preprocessing.
    If some model requires no preprocessing, the preprocessing should be defined anyways.
    Namely, preprocessing has to ensure the correct order of dimensions, and in case of PyTorch preprocessing also that PIL images get transformed to tensors.

.. caution::
    Image preprocessing functions returned by :py:meth:`~common.pipeline.pipeline.model.PreprocessingSpecs.get_tf_preprocessing_fn` should be invariant with respect to the type of the image. The possible types are `uint8 <https://www.tensorflow.org/api_docs/python/tf#uint8>`__ (intensity values 0-255) or `float32 <https://www.tensorflow.org/api_docs/python/tf#float32>`__ (`TensorFlow specification <https://www.tensorflow.org/hub/common_signatures/images#image_input>`__).

How Do the Schemas Work and How Do I Extend the Schema?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All schemas are defined with the `pydantic <https://pydantic-docs.helpmanual.io/>`__ library, so it is good to be familiar with the basics of the library before altering the schema.

Some rules, tips & tricks:

1. The ``example`` parameter is documented `in the FastAPI library <https://fastapi.tiangolo.com/tutorial/schema-extra-example/#field-additional-arguments>`__.
2. If possible, constrain possible values with built-in tools rather than with validators. Prefer using parameters of `Field function <https://pydantic-docs.helpmanual.io/usage/schema/#field-customisation>`__ to `constrained types <https://pydantic-docs.helpmanual.io/usage/types/#constrained-types>`__ as the latter are `not compatible with Mypy <https://github.com/samuelcolvin/pydantic/issues/239>`__.
3. All non-abstract pydantic models should include a `Config class <https://pydantic-docs.helpmanual.io/usage/model_config/>`__. The config should inherit from the ``_DefaultConfig`` in order to impose the constraints specified in it. Additionally, if the model is a part of a **Union type**, ``title`` field should be specified as well, so that the title is used in the generated REST documentation rather than the name of the class/model. See :ref:`duality` for an explanation of Union types.
4. When validation depends on multiple fields prefer using `root validators <https://pydantic-docs.helpmanual.io/usage/validators/#root-validators>`__ to regular validators. Root validator will be called any time that the model will change, whereas regular validator will be called only when the specified field changes. Since two fields can depend on each other we want to validate when either of them is changed, not just one of them.
5. Make sure to read `these bullet points <https://pydantic-docs.helpmanual.io/usage/validators/>`__ to understand when field values are not passed to a validator.
6. Before using string type annotations or ``from __future__ import annotations`` (Python 3.7 and above) make sure you are familiar with `limitations of pydantic and FastAPI with respect to types <https://github.com/samuelcolvin/pydantic/issues/2678>`__.
7. When copying models with ``.copy()``, make sure to use ``.copy(deep=True)`` when a deep copy is required.

.. _duality:

Duality of Types
################

There are two kinds of "supertypes" present:

1. The regular supertypes which are used with inheritance and are mostly used within code.
2. `Union type <https://docs.python.org/3/library/typing.html#typing.Union>`__, which is used to denote a choice between multiple pydantic models. In the generated REST documentation a choice is shown by displaying names of the models within a union, which are defined with ``title``.

.. attention::
    When specifying a choice between multiple models with a Union type, one has to **ensure that each model present in the union is identifiable based on the required fields**, where a `required field <https://pydantic-docs.helpmanual.io/usage/models/#required-fields>`__ simply means a field without specified default value.
    Concretely, if there are two models that have the same set of required fields, pydantic will not know which model was meant and `will rely on the order within the Union <https://pydantic-docs.helpmanual.io/usage/types/#unions>`__, which should be avoided.
    Same could happen if required fields of one model would be a subset of required fields of another model.
    However, this is prevented with the parameters specified in the ``_DefaultConfig``.

Because of the duality, additional code must be present that converts the second type to the first type, even though the underlying object stays the same.
The conversion is performed, so that the types can be checked with Mypy.

.. hint::
    Union types are denoted with a trailing **U** in code.

How Can I Use the Same ``tox`` Environment Multiple Times?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the help of `this GitHub issue <https://github.com/tox-dev/tox/issues/425>`__, I figured out that both environments should have same ``basepython`` and ``deps`` fields. Furthermore, the second environment should have ``envdir`` set to ``{toxworkdir}/<name of the first environment>``.

How Can I Add a New HuggingFace Model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In :ref:`schemas`, ``hf_name`` Literal has to be changed to include the new model. The literal is used in the first place, because preprocessing depends on the model.
2. The validator of the model should be changed to capture the new model.
3. The :py:class:`~worker_general.general.model.preprocessing.HFPreprocessing` class has to be altered to support the new model.


How Can I Log Messages?
~~~~~~~~~~~~~~~~~~~~~~~

For REST and scheduler, simply use the ``_logger`` instance.
For workers, use `get_task_logger <https://docs.celeryproject.org/en/stable/internals/reference/celery.utils.log.html#celery.utils.log.get_task_logger>`__ to obtain a logger instance.
This way each task can be identified from the logs.