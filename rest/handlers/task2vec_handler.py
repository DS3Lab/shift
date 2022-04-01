import os
import pickle

from db_tools.postgres import JobsDB
from db_tools.redis import StatusMap, Task2vecQueue
from schemas import Status
from schemas.requests.common import Task2VecRequest
from schemas.requests.task2vec import Task2VecMultipleReaderRequest
from schemas.response import StatusResponse
from task2vec.src.interface import calculate_dist_matrix

def task2vec(
        request: Task2VecMultipleReaderRequest,
        jobs_db: JobsDB,
        tq: Task2vecQueue,
        sm: StatusMap,
    ):
    successful_jobs = jobs_db.get_successful_task2vec_request_hashes()
    task2vec_requests: Task2VecRequest = request.generate_task2vec_requests()
    remaining_tasks = 0
    embeddings = []
    for request in task2vec_requests:
        if request.hash in successful_jobs:
            result_path = os.environ["RESULTS_LOCATION"]
            result_filename = os.path.join(result_path, request.hash)
            with open(result_filename, "rb") as fstream:
                embedding = pickle.load(fstream)
            embeddings.append(embedding)
        else:
            tq.put(request)
            sm[request.id] = StatusResponse(status=Status.WAITING)
            remaining_tasks += 1
    distance_matrix = calculate_dist_matrix(embeddings)
    return distance_matrix.tolist(), remaining_tasks
