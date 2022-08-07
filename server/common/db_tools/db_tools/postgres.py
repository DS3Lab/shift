import json
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import date
from typing import Dict, Optional, Sequence, Set, Tuple, Union

from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.sql import SQL, Identifier
from schemas import Hash
from schemas.dataset import DatasetRegistrationRequest
from schemas.models import (
    ImageFullModelConfig,
    TextFullModelConfig,
    get_predefined_image_model_configs_with_info,
    get_predefined_text_model_configs_with_info,
)
from schemas.models.image_model import FinetunedTFFullImageModelConfig, ImageModelInfo
from schemas.requests.common import ClassifierRequest, InferenceRequest, Task2VecRequest
from schemas.requests.finetune import FinetuneRequest
from schemas.requests.model_request import (
    ImageModelInfoRequest,
    ImageModelRegistrationRequest,
    QueryModelByTagsRequest,
    ReadersUsedWithAModelRequest,
    TextModelInfoRequest,
    TextModelRegistrationRequest,
)
from schemas.requests.reader import ModelsUsedWithAReaderRequest
from schemas.response import (
    LinearResult,
    MatchingImageModelsResponse,
    MatchingTextModelsResponse,
    ModelsUsedWithAReaderResponse,
    NearestNeighborResult,
    ReadersUsedWithAModelResponse,
)
from schemas.task.result import KnownResult

__all__ = ["JobsDBInterface", "BaseDBInterface", "JobsDB"]


@contextmanager
def _db_cursor(pool: ThreadedConnectionPool, read_only: bool = False):
    # https://www.psycopg.org/docs/pool.html
    # https://www.psycopg.org/docs/connection.html#connection.autocommit
    connection = pool.getconn()
    connection.set_session(
        isolation_level=ISOLATION_LEVEL_READ_COMMITTED,
        readonly=read_only,
        autocommit=False,
    )
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    finally:
        cursor.close()
        pool.putconn(connection)


class JobsDBInterface(ABC):
    @abstractmethod
    def store_inference_job(self, req: InferenceRequest):
        """Stores inference job/request to the database.

        Args:
            req (InferenceRequest): Inference request to store.
        """
        raise NotImplementedError

    @abstractmethod
    def store_nearest_neighbor_job(
        self, classifier_job_hash: Hash, nn_result: NearestNeighborResult
    ):
        """Stores the nearest neighbor job result.

        Args:
            classifier_job_hash (Hash): Identifier of a specific nearest neighbor job.
            nn_result (NearestNeighborResult): Result from which a new result can be computed if only labels get changed.
        """
        raise NotImplementedError

    @abstractmethod
    def store_known_result(
        self,
        job_hash: Hash,
        nn_result: Union[NearestNeighborResult, LinearResult],
        classifier_request: ClassifierRequest,
    ):
        """Store Known Results"""
        raise NotImplementedError

    @abstractmethod
    def query_model_by_tags(request: QueryModelByTagsRequest):
        """Query model by tags"""
        raise NotImplementedError

    @abstractmethod
    def store_task2vec_job(self, req: Task2VecRequest):
        raise NotImplementedError

    @abstractmethod
    def store_finetune_job(self, req: FinetuneRequest):
        """Store finetune job"""
        raise NotImplementedError

    @abstractmethod
    def get_reader_by_json(self, json_reader: str):
        raise NotImplementedError

    @abstractmethod
    def get_known_result_by_params(
        self,
        classifier_type: str,
        model_json: str,
        train_reader_json: str,
        test_reader_json: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def purge(self):
        """
        Purge saved results in database
        """
        raise NotImplementedError

    @abstractmethod
    def get_nn_result(self, nn_job_hash: Hash) -> Optional[NearestNeighborResult]:
        """Retrieves the nearest neighbor result from the database.

        Args:
            nn_job_hash (Hash): Identifier of a specific nearest neighbor job.

        Returns:
            NearestNeighborResult, optional: Result from which a new result can be
            computed if only labels get changed. The value None means that the result
            for the specific job (identified by its hash) has not been stored yet.
        """
        raise NotImplementedError

    @abstractmethod
    def get_successful_inference_request_hashes(self) -> Set[Hash]:
        """Retrieves hashes of inference jobs that successfully completed in the past.

        Returns:
            Set[Hash]: Hashes of successful inference jobs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_successful_classifier_request_hashes_and_errors(self) -> Dict[Hash, float]:
        """Retrieves hashes of classifier jobs that successfully completed in the past
        and the error they achieved.

        Returns:
            Dict[Hash, float]: Mapping from successful classifier job hashes to their
            errors.
        """
        raise NotImplementedError

    @abstractmethod
    def get_successful_task2vec_request_hashes(self) -> Set[Hash]:
        """Retrieves hashes of task2vec jobs that successfully completed in the past

        Returns:
            Set[Hash]: Hashes of successful task2vec jobs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_successful_finetune_request_hashes(self) -> Set[Hash]:
        """Retrieves hashes of finetune jobs that successfully completed in the past

        Returns:
            Set[Hash]: Hashes of successful finetune jobs
        """
        raise NotImplementedError

    @abstractmethod
    def store_linear_job(self, h: Hash, linear_result: LinearResult):
        """Store linear classification job"""
        raise NotImplementedError

    @abstractmethod
    def store_hyperband_job(self, h: Hash):
        """Store linear classification job"""
        raise NotImplementedError

    @abstractmethod
    def check_hyperband_job(self, h: Hash):
        """Store linear classification job"""
        raise NotImplementedError


class BaseDBInterface(ABC):
    @abstractmethod
    def populate_model_databases(self):
        """Populates model databases with predefined models."""
        raise NotImplementedError

    @abstractmethod
    def populate_tasks_database(self):
        """Populates tasks database with predefined tasks."""
        raise NotImplementedError

    @abstractmethod
    def register_image_model(self, request: ImageModelRegistrationRequest):
        """Stores an image model together with information about it in the database.

        Args:
            request (ImageModelRegistrationRequest): Model and information about it.
        """
        # TODO: return values from DB
        raise NotImplementedError

    @abstractmethod
    def register_text_model(self, request: TextModelRegistrationRequest):
        """Stores a text model together with information about it in the database.

        Args:
            request (TextModelRegistrationRequest): Model and information about it.
        """
        # TODO: return values from DB
        raise NotImplementedError

    @abstractmethod
    def register_reader(self, request: DatasetRegistrationRequest):
        raise NotImplementedError

    @abstractmethod
    def get_batch_size(
        self, model: Union[ImageFullModelConfig, TextFullModelConfig]
    ) -> Optional[int]:
        """Retrieves the batch size that was specified for a model.

        Args:
            model (Union[ImageFullModelConfig, TextFullModelConfig]): Model for which the batch size will be returned.

        Returns:
            int, optional: The batch size. The value None means that batch size for the model was not specified in the past.
        """
        raise NotImplementedError

    @abstractmethod
    def get_readers_used_with_a_model(
        self, request: ReadersUsedWithAModelRequest
    ) -> ReadersUsedWithAModelResponse:
        """Retrieves readers that were used in the past together with the specified model.

        Args:
            request (ReadersUsedWithAModelRequest): Model for which the corresponding readers will be returned.

        Returns:
            ReadersUsedWithAModelResponse: Readers that were used with the specified model and the hash identifying the job in which they were used together.
        """
        raise NotImplementedError

    @abstractmethod
    def get_models_used_with_a_reader(
        self, request: ModelsUsedWithAReaderRequest
    ) -> ModelsUsedWithAReaderResponse:
        """Retrieves models that were used in the past together with the specified reader.

        Args:
            request (ModelsUsedWithAReaderRequest): Reader for which the corresponding models will be returned.

        Returns:
            ModelsUsedWithAReaderResponse: Models that were used with the specified
            reader and the hash identifying the job in which they were used together.
        """
        raise NotImplementedError

    @abstractmethod
    def get_image_models(
        self, request: ImageModelInfoRequest
    ) -> MatchingImageModelsResponse:
        """Retrieves image models that satisfy the specified criteria.

        Args:
            request (ImageModelInfoRequest): Image model criteria.

        Returns:
            MatchingImageModelsResponse: Image models satisfying criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def get_text_models(
        self, request: TextModelInfoRequest
    ) -> MatchingTextModelsResponse:
        """Retrieves text models that satisfy the specified criteria.

        Args:
            request (TextModelInfoRequest): Text model criteria.

        Returns:
            MatchingTextModelResponse: Text models satisfying criteria.
        """
        raise NotImplementedError


class JobsDB(BaseDBInterface, JobsDBInterface):
    def __init__(self, dbname: str, user: str, password: str, host: str, port: int):
        self._pool = ThreadedConnectionPool(
            **{
                "dbname": dbname,
                "user": user,
                "password": password,
                "host": host,
                "port": port,
                "minconn": 20,
                "maxconn": 50,
            }
        )

    def populate_model_databases(self):
        image_model_configs = get_predefined_image_model_configs_with_info()
        text_model_configs = get_predefined_text_model_configs_with_info()

        with _db_cursor(self._pool) as cur:
            execute_values(
                cur,
                "INSERT INTO image_model "
                "(json_model, batch_size, image_size, date_added, num_params, source) "
                "VALUES %s ON CONFLICT ON CONSTRAINT unique_image_model "
                "DO NOTHING",
                [
                    (
                        model.invariant_json,
                        info.batch_size,
                        info.image_size,
                        info.date_added,
                        info.num_params,
                        model.source_str,
                    )
                    for model, info in image_model_configs
                ],
            )
            execute_values(
                cur,
                "INSERT INTO text_model "
                "(json_model, batch_size, token_length, date_added, num_params, source)"
                " VALUES %s ON CONFLICT ON CONSTRAINT unique_text_model "
                "DO NOTHING",
                [
                    (
                        model.invariant_json,
                        info.batch_size,
                        info.token_length,
                        info.date_added,
                        info.num_params,
                        model.source_str,
                    )
                    for model, info in text_model_configs
                ],
            )
    def populate_tasks_database(self):
        pass
    
    def register_image_model(self, request: ImageModelRegistrationRequest):
        model = request.full_config
        info = request.info
        # - model does not exist -> insert it
        # - model does exist -> update params that are not None/NULL
        # More info: https://www.postgresql.org/docs/current/sql-insert.html
        with _db_cursor(self._pool) as cur:
            if not request.finetuned:
                cur.execute(
                    "INSERT INTO image_model as i "
                    "(json_model, batch_size, image_size, date_added, num_params, source) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT ON CONSTRAINT unique_image_model DO UPDATE SET "
                    "batch_size = COALESCE(EXCLUDED.batch_size, i.batch_size), "
                    "image_size = COALESCE(EXCLUDED.image_size, i.image_size), "
                    "date_added = COALESCE(EXCLUDED.date_added, i.date_added), "
                    "num_params = COALESCE(EXCLUDED.num_params, i.num_params), "
                    "source = EXCLUDED.source",
                    (
                        model.invariant_json,
                        info.batch_size,
                        info.image_size,
                        info.date_added,
                        info.num_params,
                        model.source_str,
                    ),
                )
            else:
                cur.execute(
                    "INSERT INTO image_model as i "
                    "(json_model, batch_size, image_size, date_added, num_params, source, tag) "
                    "VALUES (%s, %s, %s, %s, %s, %s, '{finetuned}') "
                    "ON CONFLICT ON CONSTRAINT unique_image_model DO UPDATE SET "
                    "batch_size = COALESCE(EXCLUDED.batch_size, i.batch_size), "
                    "image_size = COALESCE(EXCLUDED.image_size, i.image_size), "
                    "date_added = COALESCE(EXCLUDED.date_added, i.date_added), "
                    "num_params = COALESCE(EXCLUDED.num_params, i.num_params), "
                    "source = EXCLUDED.source",
                    (
                        model.invariant_json,
                        info.batch_size,
                        info.image_size,
                        info.date_added,
                        info.num_params,
                        model.source_str,
                    ),
                )

    def register_text_model(self, request: TextModelRegistrationRequest):
        model = request.full_config
        info = request.info
        # - model does not exist -> insert it
        # - model does exist -> update params that are not None/NULL
        with _db_cursor(self._pool) as cur:
            cur.execute(
                "INSERT INTO text_model as t "
                "(json_model, batch_size, token_length, date_added, num_params, "
                "source) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT ON CONSTRAINT unique_text_model DO UPDATE SET "
                "batch_size = COALESCE(EXCLUDED.batch_size, t.batch_size), "
                "token_length = COALESCE(EXCLUDED.token_length, t.token_length), "
                "date_added = COALESCE(EXCLUDED.date_added, t.date_added), "
                "num_params = COALESCE(EXCLUDED.num_params, t.num_params), "
                "source = EXCLUDED.source",
                (
                    model.invariant_json,
                    info.batch_size,
                    info.token_length,
                    info.date_added,
                    info.num_params,
                    model.source_str,
                ),
            )

    def register_reader(self, request: DatasetRegistrationRequest):
        dataset = request.dataset.dict()
        dataset = {k: v for k, v in dataset.items() if v is not None}
        info = request.info
        with _db_cursor(self._pool) as cur:
            cur.execute(
                "INSERT INTO reader as t"
                "(json_dataset, date_added, size, name, path) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT ON CONSTRAINT unique_task_dataset DO UPDATE SET "
                "size=COALESCE(EXCLUDED.size, t.size), "
                "date_added=COALESCE(EXCLUDED.date_added, t.date_added), "
                "name=COALESCE(EXCLUDED.name, t.name), "
                "path=COALESCE(EXCLUDED.path, t.path)",
                (
                    json.dumps(dataset),
                    info.date_added,
                    info.size,
                    info.name,
                    info.path,
                ),
            )

    def get_batch_size(
        self, model: Union[ImageFullModelConfig, TextFullModelConfig]
    ) -> Optional[int]:
        with _db_cursor(self._pool) as cur:
            table = (
                "image_model"
                if isinstance(model, ImageFullModelConfig)
                else "text_model"
            )
            query = SQL("SELECT batch_size FROM {} where json_model = %s").format(
                Identifier(table)
            )
            cur.execute(query, (model.invariant_json,))
            result = cur.fetchone()
            return result[0] if result is not None else None

    def get_model_info(
        self, model: Union[ImageFullModelConfig, TextFullModelConfig], info_name: str
    ) -> Optional[int]:
        with _db_cursor(self._pool) as cur:
            table = (
                "image_model"
                if isinstance(model, ImageFullModelConfig)
                else "text_model"
            )
            query = SQL("SELECT {} FROM {} where json_model = %s").format(
                Identifier(info_name), Identifier(table)
            )
            cur.execute(query, (model.invariant_json,))
            result = cur.fetchone()
            return result[0] if result is not None else None

    def get_readers_used_with_a_model(
        self, request: ReadersUsedWithAModelRequest
    ) -> ReadersUsedWithAModelResponse:
        model = request.full_config
        with _db_cursor(self._pool, read_only=True) as cur:
            if isinstance(model, ImageFullModelConfig):
                cur.execute(
                    "SELECT job_hash, json_reader FROM image_job "
                    "WHERE json_model = %s",
                    (model.invariant_json,),
                )
            else:
                cur.execute(
                    "SELECT job_hash, json_reader FROM text_job "
                    "WHERE json_model = %s",
                    (model.invariant_json,),
                )
            return ReadersUsedWithAModelResponse.from_tuples(cur.fetchall())

    def get_reader_size(self, json_reader: str):
        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT size " "FROM reader where json_dataset = %s",
                (json_reader,),
            )
            result = cur.fetchone()
            return result[0] if result is not None else None

    def get_reader_by_json(self, json_reader: str):
        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT name " "FROM reader WHERE json_dataset = %s",
                (json_reader,),
            )
            result = cur.fetchone()
            return result[0] if result is not None else None

    def get_all_registered_readers(self):
        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute("SELECT name, json_dataset FROM reader;")

            return [
                {"name": row[0], "json": json.loads(row[1])} for row in cur.fetchall()
            ]

    def get_registered_readers_with_name(self, name: str):
        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT name, json_dataset FROM reader where name=%s;",
                (name,),
            )
            return [
                {"name": row[0], "json": json.loads(row[1])} for row in cur.fetchall()
            ]

    def get_models_used_with_a_reader(
        self, request: ModelsUsedWithAReaderRequest
    ) -> ModelsUsedWithAReaderResponse:
        reader_json = request.reader_config_with_checked_type.invariant_json
        with _db_cursor(self._pool, read_only=True) as cur:
            # Need two queries, we do not know the type of the reader
            cur.execute(
                "SELECT job_hash, json_model FROM image_job WHERE json_reader = %s",
                (reader_json,),
            )
            image_results = cur.fetchall()
            cur.execute(
                "SELECT job_hash, json_model FROM text_job WHERE json_reader = %s",
                (reader_json,),
            )
            text_results = cur.fetchall()
            return ModelsUsedWithAReaderResponse.from_tuples(
                image_results + text_results
            )

    def get_image_models(
        self, request: ImageModelInfoRequest
    ) -> MatchingImageModelsResponse:

        query = (
            "SELECT * from image_model WHERE "
            "(%(d_null)s OR date_added BETWEEN %(d_min)s AND %(d_max)s) AND "
            "(%(n_null)s OR num_params BETWEEN %(n_min)s AND %(n_max)s) AND "
            "(%(source)s IS NULL OR source = %(source)s) AND "
            "(%(i_null)s OR image_size BETWEEN %(i_min)s AND %(i_max)s) AND "
            "(%(di_null)s OR dimension BETWEEN %(di_min)s AND %(di_max)s) AND "
            "(%(di_null)s OR dimension BETWEEN %(di_min)s AND %(di_max)s)"
        )
        if not request.finetuned:
            query = query + " AND 'finetuned' != ALL(coalesce(tag, array[]::text[]));"
        date_added = request.date_added
        num_params = request.num_params
        image_size = request.image_size
        dimension_size = request.dimension
        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute(
                query,
                {
                    "d_null": date_added is None,
                    "d_min": date_added.min if date_added is not None else None,
                    "d_max": date_added.max if date_added is not None else None,
                    "n_null": num_params is None,
                    "n_min": num_params.min if num_params is not None else None,
                    "n_max": num_params.max if num_params is not None else None,
                    "source": request.source,
                    "i_null": image_size is None,
                    "i_min": image_size.min if image_size is not None else None,
                    "i_max": image_size.max if image_size is not None else None,
                    "di_null": dimension_size is None,
                    "di_min": dimension_size.min
                    if dimension_size is not None
                    else None,
                    "di_max": dimension_size.max
                    if dimension_size is not None
                    else None,
                },
            )
            json_sequence: Sequence[str] = [
                {"str": r[1], "num_params": r[5], "up_acc": r[9]}
                for r in cur.fetchall()
            ]
            return MatchingImageModelsResponse.from_model_json_sequence(json_sequence)

    def get_text_models(
        self, request: TextModelInfoRequest
    ) -> MatchingTextModelsResponse:
        query = (
            "SELECT * from text_model WHERE "
            "(%(d_null)s OR date_added BETWEEN %(d_min)s AND %(d_max)s) AND "
            "(%(n_null)s OR num_params BETWEEN %(n_min)s AND %(n_max)s) AND "
            "(%(source)s IS NULL OR source = %(source)s) AND "
            "(%(t_null)s OR token_length BETWEEN %(t_min)s AND %(t_max)s)"
        )
        if not request.finetuned:
            query = query + "'finetuned' != ALL(coalesce(tag, array[]::text[]));"
        date_added = request.date_added
        num_params = request.num_params
        token_length = request.token_length

        with _db_cursor(self._pool, read_only=True) as cur:
            cur.execute(
                query,
                {
                    "d_null": date_added is None,
                    "d_min": date_added.min if date_added is not None else None,
                    "d_max": date_added.max if date_added is not None else None,
                    "n_null": num_params is None,
                    "n_min": num_params.min if num_params is not None else None,
                    "n_max": num_params.max if num_params is not None else None,
                    "source": request.source,
                    "t_null": token_length is None,
                    "t_min": token_length.min if token_length is not None else None,
                    "t_max": token_length.max if token_length is not None else None,
                },
            )
            json_sequence: Sequence[str] = [r[1] for r in cur.fetchall()]
            print(json_sequence)
        return MatchingTextModelsResponse.from_model_json_sequence(json_sequence)

    def store_inference_job(self, req: InferenceRequest):
        with _db_cursor(self._pool) as cur:
            model = req.model_config_with_checked_type
            job_table = (
                "image_job" if isinstance(model, ImageFullModelConfig) else "text_job"
            )
            if hasattr(req.reader, "slice") and req.reader.slice is not None:
                query = SQL(
                    "INSERT INTO {} "
                    "(job_hash, json_reader, json_model, slice_start, slice_stop) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(Identifier(job_table))
                cur.execute(
                    query,
                    (
                        req.hash,
                        req.reader_config_with_checked_type.invariant_json,
                        model.invariant_json,
                        req.reader.slice.start,
                        req.reader.slice.stop,
                    ),
                )
            else:
                query = SQL(
                    "INSERT INTO {} "
                    "(job_hash, json_reader, json_model) "
                    "VALUES (%s, %s, %s)"
                ).format(Identifier(job_table))
                cur.execute(
                    query,
                    (
                        req.hash,
                        req.reader_config_with_checked_type.invariant_json,
                        model.invariant_json,
                    ),
                )

    def store_task2vec_job(self, req: Task2VecRequest):
        with _db_cursor(self._pool) as cur:
            cur.execute(
                "INSERT INTO task2vec_job (job_hash, json_reader, json_model) VALUES (%s, %s, %s)",
                (
                    req.hash,
                    req.reader.invariant_json,
                    req.probe.invariant_json,
                ),
            )

    def store_nearest_neighbor_job(
        self, classifier_job_hash: Hash, nn_result: NearestNeighborResult
    ):
        with _db_cursor(pool=self._pool) as cur:
            cur.execute(
                "INSERT INTO classifier_job (job_hash, test_labels, "
                "test_indices_within_readers, test_reader_indices, train_labels, "
                "train_indices_within_readers, train_reader_indices, error, raw_error) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);",
                (
                    classifier_job_hash,
                    nn_result.test_labels,
                    nn_result.test_indices_within_readers,
                    nn_result.test_reader_indices,
                    nn_result.train_labels,
                    nn_result.train_indices_within_readers,
                    nn_result.train_reader_indices,
                    nn_result.error,
                    nn_result.raw_error,
                ),
            )

    def store_linear_job(self, classifier_job_hash: Hash, linear_result: LinearResult):
        with _db_cursor(pool=self._pool) as cur:
            cur.execute(
                "INSERT INTO classifier_job (job_hash, test_labels, predicted_test_labels, error, raw_error) VALUES (%s, %s, %s, %s, %s);",
                (
                    classifier_job_hash,
                    linear_result.test_labels,
                    linear_result.predicted_test_labels,
                    linear_result.error,
                    linear_result.raw_error,
                ),
            )

    def get_info_with_inference_hash(
        self, inference_job_hash: Hash
    ) -> Tuple[str, str, Dict, Dict]:
        inference_type = "text"
        with _db_cursor(pool=self._pool) as cur:
            cur.execute(
                "SELECT job_hash, json_reader, json_model FROM text_job where job_hash = %s",
                (inference_job_hash,),
            )
            result = cur.fetchone()
            if result is None:
                # if it is not in the text_job, then try in image_job
                cur.execute(
                    "SELECT job_hash, json_reader, json_model FROM image_job where job_hash= %s ",
                    (inference_job_hash,),
                )
                result = cur.fetchone()
                inference_type = "image"
            return (inference_type, *result)

    def get_known_result(self, classifier_hash: Hash):
        results = []
        with _db_cursor(self._pool) as cur:
            cur.execute(
                "SELECT model_json, classifier_job_hash, classifier_type, value from known_result WHERE classifier_job_hash=%s;",
                (classifier_hash,),
            )
            results = cur.fetchall()
        return [
            KnownResult(
                json_model=json.loads(each[0]),
                classify_job_hash=each[1],
                classifier=each[2],
                err=each[3],
            )
            for each in results
        ]

    def get_known_result_by_params(
        self,
        classifier_type,
        model_json,
        train_reader_json,
        test_reader_json,
    ):
        cleaned_readers = []
        for test_reader in test_reader_json:
            cleaned_reader = {
                k: v for k, v in json.loads(test_reader).items() if v is not None
            }
            cleaned_reader = json.dumps(cleaned_reader)
            cleaned_readers.append(cleaned_reader)
        test_reader_json = cleaned_readers
        with _db_cursor(self._pool) as cur:
            cur.execute(
                "SELECT model_json, classifier_job_hash, classifier_type, value from known_result WHERE classifier_type=%s AND model_json=%s AND train_readers_json=%s AND test_readers_json=%s;",
                (classifier_type, model_json, train_reader_json, test_reader_json),
            )
            result = cur.fetchone()
        if result is not None:
            return KnownResult(
                json_model=json.loads(result[0]),
                classify_job_hash=result[1],
                classifier=result[2],
                err=result[3],
            )
        else:
            return None

    def store_known_result(
        self,
        job_hash: Hash,
        nn_result: NearestNeighborResult,
        classifier_request: ClassifierRequest,
    ):
        with _db_cursor(pool=self._pool) as cur:
            train_jobs = [
                self.get_info_with_inference_hash(train.inference_request_hash)
                for train in classifier_request.train
            ]

            test_jobs = [
                self.get_info_with_inference_hash(test.inference_request_hash)
                for test in classifier_request.test
            ]

            train_readers_json = [train_job[2] for train_job in train_jobs]
            test_readers_json = [test_job[2] for test_job in test_jobs]
            # the train_jobs here should have the same job_type, model_json, but different reader_json
            example_train_job = self.get_info_with_inference_hash(
                classifier_request.train[0].inference_request_hash
            )
            result = nn_result.error
            cur.execute(
                "INSERT INTO known_result (job_type, classifier_type, model_json, train_readers_json, test_readers_json, classifier_job_hash, value) values (%s, %s, %s, %s, %s, %s, %s)",
                (
                    example_train_job[0],
                    classifier_request.classifier.value,
                    example_train_job[3],
                    train_readers_json,
                    test_readers_json,
                    job_hash,
                    result,
                ),
            )

    def get_nn_result(self, nn_job_hash: Hash) -> Optional[NearestNeighborResult]:
        result = None
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT test_labels, test_indices_within_readers, test_reader_indices, "
                "train_labels, train_indices_within_readers, train_reader_indices "
                "FROM classifier_job WHERE job_hash = %s",
                (nn_job_hash,),
            )
            result = cur.fetchone()
            if result is None:
                return None
            return NearestNeighborResult(
                test_labels=result[0],
                test_indices_within_readers=result[1],
                test_reader_indices=result[2],
                train_labels=result[3],
                train_indices_within_readers=result[4],
                train_reader_indices=result[5],
            )

    def get_linear_result(self, lc_job_hash: Hash) -> Optional[LinearResult]:
        result = None
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT test_labels, predicted_test_labels, classifier_job FROM classifier_job WHERE job_hash = %s",
                (lc_job_hash,),
            )
            result = cur.fetchone()
            if result is None:
                return None
            return LinearResult(
                test_labels=result[0],
                predicted_test_labels=result[1],
            )

    def get_dataset(self, dataset_name: str):
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute(
                "SELECT json_dataset from reader where name=%s", (dataset_name,)
            )
            result = cur.fetchone()
            return json.loads(result[0]) if result is not None else None

    def get_all_readers(self):
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            query = SQL("SELECT name, json_dataset from reader")
            cur.execute(query)
            result = cur.fetchall()
            return result

    def get_successful_inference_request_hashes(self) -> Set[Hash]:
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("SELECT job_hash FROM image_job")
            image_hashes = cur.fetchall()
            cur.execute("SELECT job_hash from text_job")
            text_hashes = cur.fetchall()
        return set([i[0] for i in image_hashes + text_hashes])

    def get_successful_classifier_request_hashes_and_errors(self) -> Dict[Hash, float]:
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("SELECT job_hash, error FROM classifier_job")
            result = cur.fetchall()
        return {h: e for h, e in result}

    def get_successful_task2vec_request_hashes(self) -> Set[Hash]:
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("select job_hash FROM task2vec_job")
            task2vec_hashes = cur.fetchall()
        return set([i[0] for i in task2vec_hashes])

    def store_hyperband_job(self, h: Hash):
        with _db_cursor(pool=self._pool) as cur:
            cur.execute("INSERT INTO task (job_hash, status) VALUES (%s, %s)", (h, 1))

    def store_finetune_job(self, req: FinetuneRequest):
        # TODO: check if text/image model
        # TODO: check the model source (tf, pytorch, hugging face, etc..)
        model_batch_size = self.get_batch_size(req.model)
        if model_batch_size is None:
            raise ValueError("Batch size for base model cannot be found...")
        num_params = self.get_model_info(req.model, "num_params")
        if num_params is None:
            # TODO: maybe set a default value? and warn the users
            raise ValueError("Number of Params for base model cannot be found...")
        model_info = ImageModelInfo(
            batch_size=model_batch_size,
            date_added=date.today(),
            num_params=num_params,
            image_size=req.model.required_image_size.height,
        )
        model_path = os.path.join(os.environ["TFHUB_CACHE_DIR"], req.hash)
        registration_req = ImageModelRegistrationRequest(
            # as we now always work with tf-2,
            # optional_tf2_output_key_field is removed
            model=FinetunedTFFullImageModelConfig(
                tf_image_model_url=model_path,
                base_model=req.model,
                train_readers=req.readers,
                output_key=req.model.output_key,
                required_image_size=req.model.required_image_size,
                lr=req.lr,
                epochs=req.epochs,
            ),
            info=model_info,
            finetuned=True,
        )
        self.register_image_model(registration_req)
        # now register to the finetunejob table
        with _db_cursor(pool=self._pool) as cur:
            cur.execute("INSERT INTO finetune_job (job_hash) values (%s)", (req.hash,))

    def get_successful_finetune_request_hashes(self) -> Set[Hash]:
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("select job_hash from finetune_job")
            finetune_hashes = cur.fetchall()
        return set([i[0] for i in finetune_hashes])

    def check_hyperband_job(self, h: Hash):
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("select * FROM task where job_hash=%s", (h,))
            result = cur.fetchone()
        return {"hash": result[0], "status": result[1]} if result is not None else None

    def purge(self):
        with _db_cursor(pool=self._pool) as cur:
            cur.execute(
                "truncate table known_result, classifier_job, image_job, text_job, finetune_job, task2vec_job, task"
            )

    def query_model_by_tags(self, request: QueryModelByTagsRequest):
        tags_string = str(request.tags)
        tags_string.replace("[", "{")
        tags_string.replace("]", "}")
        with _db_cursor(pool=self._pool, read_only=True) as cur:
            cur.execute("select json_model from {} WHERE tag && '{}';").format(
                Identifier(request.source), tags_string
            )
            json_sequence: Sequence[str] = [r[0] for r in cur.fetchall()]
        return MatchingTextModelsResponse.from_model_json_sequence(json_sequence)
