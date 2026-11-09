# Sample corpus used by the smoke test and for a first manual ingestion.

# What this project is

This project is a retrieval-augmented generation (RAG) pipeline built on LanceDB. It
ingests Markdown and plain text files, splits them into small overlapping chunks, embeds
each chunk with the `BAAI/bge-small-en-v1.5` ONNX model, and stores the vectors in a
LanceDB table together with a full-text index.

# How retrieval works

A question is embedded with the same model. The store then runs a hybrid search that
combines vector similarity with LanceDB's native full-text search, and merges the two
ranked lists with reciprocal rank fusion. The top chunks are placed in the prompt for an
OpenAI-compatible chat model, which answers with bracketed citations such as [1].

# Deployment

The pipeline runs on a local Kubernetes cluster provided by Colima. Skaffold builds three
container images and deploys a Helm chart. The LanceDB store is a StatefulSet with a
persistent volume, the FastAPI backend is a stateless Deployment, and the Streamlit chat
interface is another Deployment. Images are pushed to a small registry running inside the
Colima virtual machine at 127.0.0.1:5001.

# Why the store is separate

LanceDB is an embedded library, not a server. On a local filesystem there is no commit
arbitration between concurrent writers, so exactly one process may write the database.
The store service owns that directory, runs a single worker, and serializes writes behind
a lock. The backend and the UI never open LanceDB themselves, which keeps them stateless
and safe to scale or restart independently.
