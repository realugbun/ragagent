# Overview

This project is a RAG solution made to be used with LLMs.

It has two services an API and a Worker. The API handles various requests to search the vector db by similarity using cosign distance. It also supports various other CRUD operations like finding all chunks for a document, searching by tags, updating tags on a chunk, and deleting all chunks for a document.

The API accepts text inputs and file uploads. It validates the request and puts the job on a queue. The api returns a job id which you can query to get the status using the v1/job/:job_id endpoint.

The Worker pulls the job from the queue, encodes the text to vectors, autogenerates tags, inserts records into the database and updates the job status.

The embeddings are saved in a PostgreSQL database with the pgvector extension.

The queue is managed by Redis and rq. The AI models run close to the metal and don't handel forking well so we use a simple queue that runs on a single thread. We can scale up by creating more instances of the worker service.

# Project Setup

This project is managed with uv which saves everything in .venv

```sh
pip install uv
source .venv/bin/activate
uv sync
```

# Database

This project uses PostgreSQL with the pgvector extension. The easiest way to get a local copy running is to pull a prebuilt docker container.

```sh
docker run --name pgvector-container -e POSTGRES_USER=ragagent -e POSTGRES_PASSWORD=ragagent -e POSTGRES_DB=ragagent -p 5432:5432 -d pgvector/pgvector:pg17
```

You will need to enable the extension before the first run.

```sql
CREATE EXTENSION vector;
```

# Migrations

We use yoyo for database migrations. It uses simple SQL 

You can configure the settings including database URI in yoyo.ini

Create migration

```sh
yoyo new --sql
```

You need to manually create the rollback file with the same name as the generated file and the extension `.rollback.sql` 

List status of migrations
```sh
yoyo list
```

Apply migrations
```sh
yoyo apply
```
By default apply will prompt you to approve each migration you can skip this by passing `--batch` or setting batch to `on` in yoyo.ini

# Spacy

You need to download Spacy models directly before launching the app.

`spacy download en_core_web_md`

# Launching the app

## API

```sh
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Worker

You can use the docker compose file to start a local instance of redis.

```sh
docker compose up redis -d
```

In a terminal you need to start rq to pick up the redis queues.
```sh
uv run rq worker ${REDIS_QUEUE} --url redis://:${REDIS_PASSWORD}@redis:${REDIS_PORT}/${REDIS_DB}
```

rq will invoke the code in the worker.py file.

# Building with docker

spacy does not have a hook for Linux ARM so you must set the build for x86

```sh
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker compose build --no-cache
docker compose up -d
```

The containers take some time to launch the first time. They download an 8gb LLM.

