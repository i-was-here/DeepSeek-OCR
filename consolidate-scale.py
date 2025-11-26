#!/usr/bin/env python3
"""
Consolidate per-page OCR txt files into single markdown documents.

The script scans an S3 bucket/prefix that holds page images and their matching
`.txt` outputs (e.g. `images/<doc>/page_1.jpg` + `images/<doc>/page_1.txt`).
Whenever a document folder has txt files for every image, all txt contents are
concatenated (ordered by page number) and uploaded as
`<output-prefix>/<doc>.md`. Documents with missing txt files are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Pinecone / embedding configuration (mirrors sample-server.py)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "default-index-v2")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "768"))

if GEMINI_API_KEY and PINECONE_API_KEY:
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/embedding-001",
        api_key=GEMINI_API_KEY,
        embedding_dim=PINECONE_DIMENSION,
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pc = None  # type: ignore[assignment]

# Azure OpenAI configuration for CTD analysis (aligned with sample-server.py)
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def join_s3_path(*parts: str) -> str:
    cleaned: List[str] = []
    for part in parts:
        if not part:
            continue
        stripped = part.strip("/")
        if stripped:
            cleaned.append(stripped)
    return "/".join(cleaned)


def setup_logger(verbose: bool) -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("consolidate-scale")
    logger.setLevel(level if verbose else logging.INFO)
    return logger


def build_s3_client():
    max_pool = env_int("AWS_MAX_POOL_CONNECTIONS", 64)
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
        config=Config(
            max_pool_connections=max_pool,
            retries={"max_attempts": env_int("AWS_MAX_RETRIES", 5), "mode": "standard"},
            tcp_keepalive=True,
        ),
    )


def setup_pinecone_index():
    """
    Create or retrieve the Pinecone index used for document embeddings.
    Mirrors the logic from sample-server.py.
    """
    if pc is None:
        raise RuntimeError("Pinecone is not configured; set GEMINI_API_KEY and PINECONE_API_KEY")

    existing = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        logging.getLogger("consolidate-scale").info(
            "Using existing Pinecone index: %s", PINECONE_INDEX_NAME
        )
        return pc.Index(PINECONE_INDEX_NAME)

    logging.getLogger("consolidate-scale").info(
        "Creating new Pinecone index: %s", PINECONE_INDEX_NAME
    )
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    return pc.Index(PINECONE_INDEX_NAME)


def analyze_ctd_sections(page_texts: List[str]) -> Dict[str, Any]:
    """
    Synchronously analyze CTD dossier sections for a consolidated document,
    following the same prompt and response format as sample-server.py.
    """
    logger = logging.getLogger("consolidate-scale")
    logger.info("Running CTD analysis for %d page(s)...", len(page_texts))

    combined_text = "\n\n--- Page Break ---\n\n".join(page_texts)

    # Truncate like sample-server (first 100k characters, 50k in prompt body)
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000] + "\n\n[Text truncated...]"

    # Locate ctd-dossier-toc.txt similarly to sample-server.py
    dossier_classes_path = os.path.join(os.path.dirname(__file__), "ctd-dossier-toc.txt")
    if not os.path.exists(dossier_classes_path):
        dossier_classes_path = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "ctd-dossier-toc.txt")

    with open(dossier_classes_path, "r") as f:
        dossier_classes = f.read()

    prompt = f"""Analyze the following text extracted from a document and determine which sections of a CTD (Common Technical Document) dossier would use this content.

For each relevant section, provide:
1. The number/identifier of the sections that will use the content of the text extracted from the images. Make sure the section number/identifier is exactly as it is in the CTD dossier sections list.
2. A brief explanation of why this content is relevant to each of these sections.

The summary and title of the entire document should be at the end of the response.

Return your response as a JSON object with the following structure:
{{
    "relevant_sections": [
        {{
            "section": "Section Number like 2.3.S.1 or 2.3.P.2",
            "relevance": "Explanation of why this content is relevant"
        }},
        {{
            "section": "Section Number like 2.3.S.1 or 2.3.P.2",
            "relevance": "Explanation of why this content is relevant"
        }},
        ...
    ],
    "summary": "Brief summary of the document content",
    "title": "Title of the document"
}}

CTD dossier sections include:
{dossier_classes}

Text to analyze:
""" + combined_text[:50000]

    response = openai_client.chat.completions.create(
        model="gpt-5-chat",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in pharmaceutical regulatory documentation and CTD dossier structure.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )

    analysis = json.loads(response.choices[0].message.content)
    logger.info(
        "CTD analysis completed. Found %d relevant sections",
        len(analysis.get("relevant_sections", [])),
    )
    return analysis


def natural_page_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 0, name


def list_documents(s3_client, bucket: str, prefix: str):
    prefix = prefix.strip("/")
    search_prefix = f"{prefix}/" if prefix else ""
    paginator = s3_client.get_paginator("list_objects_v2")

    docs = defaultdict(lambda: {"images": set(), "texts": {}})

    for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if not key.lower().endswith(IMAGE_EXTENSIONS + (".txt",)):
                continue
            doc_dir = os.path.dirname(key)
            stem = os.path.splitext(os.path.basename(key))[0]
            if key.lower().endswith(".txt"):
                docs[doc_dir]["texts"][stem] = key
            else:
                docs[doc_dir]["images"].add(stem)

    return docs


def download_txt(s3_client, bucket: str, key: str) -> str:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8", errors="replace")
    return body.strip()


def compute_doc_relative_and_key(
    image_prefix: str,
    output_prefix: str,
    doc_dir: str,
) -> Tuple[str, str]:
    """
    Given the base image prefix and a document directory, compute the
    relative document name and its consolidated markdown key under
    output_prefix.
    """
    image_prefix = image_prefix.strip("/")
    base_prefix = image_prefix
    doc_relative = (
        doc_dir[len(base_prefix) + 1 :] if base_prefix and doc_dir.startswith(f"{base_prefix}/") else doc_dir
    )
    output_key = join_s3_path(
        output_prefix,
        f"{doc_relative or os.path.basename(doc_dir) or 'document'}.md",
    )
    return doc_relative, output_key


def index_document_with_pinecone(
    bucket: str,
    doc_relative: str,
    output_key: str,
    page_texts: List[str],
    user_id: str,
    organization_id: str,
    ctd_sections: str,
    summary: str,
    title: str,
) -> List[str]:
    """
    Index a consolidated document into Pinecone using LlamaIndex, following the
    pattern of update_pinecone_index_with_llamaindex in sample-server.py.
    """
    logger = logging.getLogger("consolidate-scale")
    if pc is None or not GEMINI_API_KEY or not PINECONE_API_KEY:
        logger.warning(
            "Skipping Pinecone indexing for %s because GEMINI_API_KEY or PINECONE_API_KEY is not set",
            doc_relative or output_key,
        )
        return []

    pinecone_index = setup_pinecone_index()

    # Build documents and IDs
    file_id = doc_relative or os.path.basename(output_key) or "document"

    documents: List[Document] = []
    vector_ids: List[str] = []
    filename = os.path.basename(output_key) or (doc_relative or "document")
    processed_at = datetime.utcnow().isoformat()

    for i, text in enumerate(page_texts):
        vector_id = f"{file_id}_{user_id}_{i:05d}"
        vector_ids.append(vector_id)
        metadata: Dict[str, Any] = {
            "file_id": file_id,
            "user_id": user_id,
            "organization_id": organization_id,
            "page_number": i + 1,
            "text_preview": (text or "")[:200],
            "s3_bucket": bucket,
            "s3_key": output_key,
            "indexed_at": processed_at,
            "source": "consolidate-scale",
            # Fields aligned with OG metadata
            "filename": filename,
            "content_type": "text/markdown",
            "processed_at": processed_at,
            "ctd_sections": ctd_sections,
            "summary": summary,
            "title": title,
            "markdown_key": output_key,
        }
        documents.append(
            Document(
                text=text,
                metadata=metadata,
                id_=vector_id,
            )
        )

    logger.info(
        "Indexing %d page(s) for %s into Pinecone index %s",
        len(documents),
        output_key,
        PINECONE_INDEX_NAME,
    )

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Try to attach to existing vector store, or create new
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        for doc in documents:
            index.insert(doc)
        logger.info("Inserted %d documents into existing vector store index", len(documents))
    except Exception:
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info("Created new vector store index with %d documents", len(documents))

    return vector_ids


def consolidate_document(
    s3_client,
    bucket: str,
    image_prefix: str,
    output_prefix: str,
    doc_dir: str,
    doc_assets: Dict,
    dry_run: bool = False,
    user_id: str | None = None,
    organization_id: str | None = None,
):
    doc_relative, output_key = compute_doc_relative_and_key(
        image_prefix=image_prefix, output_prefix=output_prefix, doc_dir=doc_dir
    )
    image_stems = doc_assets["images"]
    text_map = doc_assets["texts"]

    missing = [stem for stem in image_stems if stem not in text_map]
    if missing:
        logging.getLogger("consolidate-scale").warning(
            "Skipping %s (missing %d txt files)", doc_relative or doc_dir, len(missing)
        )
        return False

    logger = logging.getLogger("consolidate-scale")
    ordered_stems = sorted(image_stems, key=natural_page_key)

    combined_parts: List[str] = []

    def _fetch(stem: str):
        txt_key = text_map[stem]
        try:
            content = download_txt(s3_client, bucket, txt_key)
            return stem, content
        except ClientError as exc:
            logger.error(
                "Failed to download %s for %s: %s", txt_key, doc_relative or doc_dir, exc
            )
            return None

    max_workers = min(len(ordered_stems) or 1, env_int("CONSOLIDATE_DOWNLOAD_WORKERS", 16))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_fetch, ordered_stems):
            if result is None:
                return False
            _, content = result
            combined_parts.append(content)

    markdown_content = "\n\n".join([part for part in combined_parts if part]).strip() + "\n"

    # Infer CTD sections, summary, and title using the same logic as sample-server.py
    ctd_sections_str = ""
    summary = ""
    title = ""
    try:
        analysis = analyze_ctd_sections(combined_parts)
        sections = analysis.get("relevant_sections", []) or []
        section_ids = [s.get("section") for s in sections if s.get("section")]
        if section_ids:
            ctd_sections_str = ";" + ";".join(section_ids) + ";"
        summary = analysis.get("summary", "") or ""
        title = analysis.get("title", "") or ""
    except Exception as exc:
        logger.error(
            "CTD analysis failed for %s: %s", doc_relative or doc_dir, exc
        )

    if dry_run:
        return True

    s3_client.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=markdown_content.encode("utf-8"),
        ContentType="text/markdown",
    )

    # Also index the document into Pinecone using per-page texts
    try:
        index_document_with_pinecone(
            bucket=bucket,
            doc_relative=doc_relative,
            output_key=output_key,
            page_texts=combined_parts,
            user_id=user_id or "system",
            organization_id=organization_id or "system",
            ctd_sections=ctd_sections_str,
            summary=summary,
            title=title,
        )
    except Exception as exc:
        logging.getLogger("consolidate-scale").error(
            "Failed to index %s into Pinecone: %s", output_key, exc
        )

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate OCR txt files into markdown documents.")
    parser.add_argument("bucket", help="S3 bucket containing the image + txt assets")
    parser.add_argument("--prefix", help="Prefix where image folders live (e.g., images)", default="images")
    parser.add_argument(
        "--output-prefix",
        default=os.getenv("CONSOLIDATE_OUTPUT_PREFIX", "markdowns"),
        help="Prefix where combined markdowns should be uploaded",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Stop after processing this many document folders",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="system",
        help="User ID to attach to Pinecone metadata (default: system)",
    )
    parser.add_argument(
        "--organization-id",
        type=str,
        default="system",
        help="Organization ID to attach to Pinecone metadata (default: system)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not upload, just log actions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)
    bucket = args.bucket
    prefix = args.prefix.strip("/")
    output_prefix = args.output_prefix.strip("/")
    user_id = args.user_id
    organization_id = args.organization_id

    s3 = build_s3_client()
    docs = list_documents(s3, bucket, prefix)
    if not docs:
        logger.info("No assets found under s3://%s/%s", bucket, prefix)
        return

    logger.info("Found %d document folders under s3://%s/%s", len(docs), bucket, prefix)

    processed = 0
    successes = 0
    for doc_dir, assets in docs.items():
        if args.limit_docs is not None and processed >= args.limit_docs:
            break
        processed += 1
        # Skip if consolidated markdown already exists
        doc_relative, output_key = compute_doc_relative_and_key(
            image_prefix=prefix, output_prefix=output_prefix, doc_dir=doc_dir
        )
        try:
            s3.head_object(Bucket=bucket, Key=output_key)
            logger.info(
                "Skipping %s because consolidated markdown already exists at s3://%s/%s",
                doc_relative or doc_dir,
                bucket,
                output_key,
            )
            continue
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in ("404", "NoSuchKey", "NotFound"):
                logger.error(
                    "Error checking existence of %s for %s: %s",
                    output_key,
                    doc_relative or doc_dir,
                    exc,
                )
                continue

        ok = consolidate_document(
            s3_client=s3,
            bucket=bucket,
            image_prefix=prefix,
            output_prefix=output_prefix,
            doc_dir=doc_dir,
            doc_assets=assets,
            dry_run=args.dry_run,
            user_id=user_id,
            organization_id=organization_id,
        )
        if ok:
            successes += 1

    logger.info("Consolidated %d/%d documents into %s", successes, processed, output_prefix or "(root)")


if __name__ == "__main__":
    main()


