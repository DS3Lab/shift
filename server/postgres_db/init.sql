DROP TABLE IF exists finetune_job, task2vec_job , image_job, text_job, classifier_job, image_model, text_model, text_job, classifier_job, dataset;


CREATE TABLE image_model (
    id serial,
    json_model text CONSTRAINT unique_image_model PRIMARY KEY,
    batch_size bigint,
    image_size bigint,
    date_added date,
    num_params bigint,
    dimension bigint,
    source text,
    tag text[],
    up_acc double precision
);

CREATE TABLE text_model (
    id serial,
    json_model text CONSTRAINT unique_text_model PRIMARY KEY,
    batch_size bigint,
    token_length bigint,
    date_added date,
    num_params bigint,
    source text,
    tag text[],
    dimension bigint,
    up_acc double precision
);

CREATE TABLE image_job (
    job_hash text CONSTRAINT unique_image_job PRIMARY KEY,
    json_reader text,
    json_model text REFERENCES image_model(json_model),
    slice_start bigint,
    slice_stop bigint
);

CREATE TABLE text_job (
    job_hash text CONSTRAINT unique_text_job PRIMARY KEY,
    json_reader text,
    json_model text REFERENCES text_model(json_model),
    slice_start bigint,
    slice_stop bigint
);

CREATE TABLE classifier_job (
    job_hash text CONSTRAINT unique_classifier_job PRIMARY KEY,
    test_labels bigint[],
    predicted_test_labels bigint[],
    test_indices_within_readers bigint[],
    test_reader_indices bigint[],
    train_labels bigint[],
    train_indices_within_readers bigint[],
    train_reader_indices bigint[],
    error double precision,
    raw_error bigint
);

CREATE TABLE reader (
    id serial,
    json_dataset text CONSTRAINT unique_task_dataset PRIMARY KEY,
    date_added date,
    size bigint,
    name text,
    path text
);

CREATE TABLE known_result (
    job_type text,
    classifier_type text,
    model_json text,
    train_readers_json text[],
    test_readers_json text[],
    classifier_job_hash text,
    value double precision
);

CREATE TABLE task2vec_job (
    job_hash text CONSTRAINT unique_task2vec_job PRIMARY KEY,
    json_reader text,
    json_model text REFERENCES image_model(json_model)
);

CREATE TABLE task (
    job_hash text,
    status bigint
);

CREATE TABLE finetune_job (
    job_hash text CONSTRAINT unique_finetune_job PRIMARY KEY,
    json_reader text[],
    json_text_model text REFERENCES text_model(json_model),
    json_image_model text REFERENCES image_model (json_model)
);