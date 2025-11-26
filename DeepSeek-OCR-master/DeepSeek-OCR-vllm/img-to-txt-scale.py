#!/usr/bin/env python3
"""
Run DeepSeek-OCR in batch mode over page images stored in S3.

Given a bucket/prefix that contains page images (e.g., page_{N}.jpg), this script
enumerates every matching object, downloads them in mini-batches, performs OCR
with the vLLM-based DeepSeek model, and uploads per-page .txt outputs back to S3
so they sit alongside the original images.

Example:
    ./img-to-txt-scale.py my-bucket images/

Environment variables (override CLI defaults):
    IMG_TXT_OUTPUT_PREFIX       (default: inherit source prefix)
    IMG_TXT_MINI_BATCH_SIZE     (default: 12)
    AWS_*                       (standard AWS credentials + region)
    AWS_MAX_POOL_CONNECTIONS    (default: 64)
    AWS_MAX_RETRIES             (default: 5)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, Iterable, List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PIL import Image
from tqdm.auto import tqdm

# Ensure DeepSeek-OCR-vllm modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VLLM_DIR = os.path.join(PROJECT_ROOT, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)

import torch  # noqa: E402,F401
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from deepseek_ocr import DeepseekOCRForCausalLM  # type: ignore  # noqa: E402
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # type: ignore  # noqa: E402
from process.image_process import DeepseekOCRProcessor  # type: ignore  # noqa: E402
from config import MODEL_PATH, PROMPT, CROP_MODE, MAX_CONCURRENCY  # type: ignore  # noqa: E402

load_dotenv()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


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
    logger = logging.getLogger("img-to-txt-scale")
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if verbose:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logger.disabled = False
    else:
        logger.disabled = True
    return logger


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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


def list_image_keys(
    s3_client,
    bucket: str,
    prefix: str,
    output_prefix: str,
) -> List[str]:
    """
    Return image keys under the given prefix **only** when the corresponding
    .txt object (as computed by `upload_texts_for_keys`) does not already exist
    under `output_prefix`.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    normalized_prefix = prefix.strip("/")
    search_prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    # First, collect all image keys under the source prefix.
    image_keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if key.lower().endswith(IMAGE_EXTENSIONS):
                image_keys.append(key)

    # Then, collect all existing txt keys under the output prefix so that we can
    # skip images that have already been processed.
    out_normalized = output_prefix.strip("/")
    out_search_prefix = f"{out_normalized}/" if out_normalized else ""
    existing_txt_keys = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=out_search_prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if key.lower().endswith(".txt"):
                existing_txt_keys.add(key)

    base_prefix = prefix.strip("/")
    filtered_keys: List[str] = []
    for key in image_keys:
        relative = (
            key[len(base_prefix) + 1 :]
            if base_prefix and key.startswith(f"{base_prefix}/")
            else key
        )
        rel_dir = os.path.dirname(relative)
        stem = os.path.splitext(os.path.basename(relative))[0]
        text_key = join_s3_path(output_prefix, rel_dir, f"{stem}.txt")
        if text_key not in existing_txt_keys:
            filtered_keys.append(key)

    filtered_keys.sort()
    return filtered_keys


def fetch_batch_images(
    s3_client,
    bucket: str,
    keys: List[str],
) -> List[Tuple[str, bytes]]:
    def _download(key: str) -> Tuple[str, bytes]:
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        return key, buffer.getvalue()

    max_workers = min(len(keys) or 1, env_int("S3_DOWNLOAD_WORKERS", 16))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_download, keys),
                total=len(keys),
                desc="Downloading batch (RAM)",
                leave=False,
            )
        )

    return results


def build_llm_and_params():
    logger = logging.getLogger("img-to-txt-scale")
    logger.info("Initializing DeepSeek OCR model & vLLM ...")
    if torch.version.cuda == "11.8":
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.96,
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    logger.info("Model ready.")
    return llm, sampling_params


def build_batch_inputs(image_payloads: List[Tuple[str, bytes]]) -> List[Dict]:
    processor = DeepseekOCRProcessor()
    inputs: List[Dict] = []
    for _, payload in image_payloads:
        image = Image.open(BytesIO(payload)).convert("RGB")
        try:
            cache_item = {
                "prompt": PROMPT,
                "multi_modal_data": {
                    "image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
                },
            }
            inputs.append(cache_item)
        finally:
            try:
                image.close()
            except Exception:
                pass
    return inputs


def clean_formula(text: str) -> str:
    formula_pattern = r"\\\[(.*?)\\\]"

    def _process(match):
        formula = match.group(1)
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", formula)
        return r"\\[" + formula.strip() + r"\\]"

    return re.sub(formula_pattern, _process, text, flags=re.DOTALL)


def strip_ref_det_blocks(text: str) -> str:
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    for full, _, _ in matches:
        text = text.replace(full, "")
    text = (
        text.replace("\n\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
        .replace("<center>", "")
        .replace("</center>", "")
    )
    return text


def run_llm_on_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_payloads: List[Tuple[str, bytes]],
) -> Dict[str, str]:
    """
    Tokenize a batch of in-memory images with DeepseekOCRProcessor, run the
    vLLM engine, and normalize the textual outputs.
    """

    processor = DeepseekOCRProcessor()

    def _prepare(
        kv: Tuple[str, bytes]
    ) -> Tuple[str, Dict[str, Dict[str, List[torch.Tensor]]]]:
        key, payload = kv
        image = Image.open(BytesIO(payload)).convert("RGB")
        try:
            tokens = processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
        finally:
            try:
                image.close()
            except Exception:
                pass
        cache_item = {
            "prompt": PROMPT,
            "multi_modal_data": {"image": tokens},
        }
        return key, cache_item

    max_workers = min(len(batch_payloads) or 1, env_int("IMG_TXT_PREPROC_WORKERS", 16))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prepared_items = list(executor.map(_prepare, batch_payloads))

    inputs = [item for _, item in prepared_items]
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results: Dict[str, str] = {}
    for (key, _), output in zip(prepared_items, outputs):
        content = output.outputs[0].text
        content = clean_formula(content)
        content = strip_ref_det_blocks(content)
        results[key] = content.strip()
    return results


def upload_texts_for_keys(
    s3_client,
    bucket: str,
    output_prefix: str,
    base_prefix: str,
    texts_by_key: Dict[str, str],
):
    base_prefix = base_prefix.strip("/")
    upload_items: List[Tuple[str, str]] = []
    for key, content in texts_by_key.items():
        relative = (
            key[len(base_prefix) + 1 :]
            if base_prefix and key.startswith(f"{base_prefix}/")
            else key
        )
        rel_dir = os.path.dirname(relative)
        stem = os.path.splitext(os.path.basename(relative))[0]
        text_key = join_s3_path(output_prefix, rel_dir, f"{stem}.txt")
        upload_items.append((text_key, content))

    def _put(item: Tuple[str, str]):
        key, content = item
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=(content + "\n").encode("utf-8"),
            ContentType="text/plain",
        )
        return key

    max_workers = min(len(upload_items) or 1, env_int("S3_UPLOAD_WORKERS", 16))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_put, upload_items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert S3 page images into txt files via DeepSeek-OCR.")
    parser.add_argument("bucket", help="S3 bucket containing the page images")
    parser.add_argument("--prefix", help="Prefix inside the bucket where images live", default="images")
    parser.add_argument(
        "--output-prefix",
        default=os.getenv("IMG_TXT_OUTPUT_PREFIX"),
        help="Root prefix for text uploads (default: reuse the image prefix)",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=env_int("IMG_TXT_MINI_BATCH_SIZE", 200),
        help="Number of pages per LLM batch",
    )
    parser.add_argument("--limit-images", type=int, default=None, help="Process at most N images from the prefix")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging", default=False)
    return parser.parse_args()


def summarize(stats: Dict[str, int]):
    logger = logging.getLogger("img-to-txt-scale")
    logger.info("===== Summary =====")
    logger.info("Batches processed: %d", stats["batches"])
    logger.info("Images processed: %d", stats["images"])
    if stats.get("errors"):
        logger.warning("Batches with errors: %d", stats["errors"])


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)
    bucket = args.bucket
    prefix = args.prefix.strip("/")
    output_bucket = bucket
    output_prefix = (
        args.output_prefix.strip("/")
        if args.output_prefix is not None
        else prefix
    )

    s3 = build_s3_client()
    image_keys = list_image_keys(s3, bucket, prefix, output_prefix)
    if args.limit_images is not None:
        image_keys = image_keys[: args.limit_images]
    if not image_keys:
        logger.info("No image objects found under s3://%s/%s", bucket, prefix)
        return

    logger.info("Found %d page images under s3://%s/%s", len(image_keys), bucket, prefix)

    llm, sampling_params = build_llm_and_params()

    stats = {"batches": 0, "images": 0, "errors": 0}
    for batch_index, batch_keys in enumerate(chunked(image_keys, args.mini_batch_size), start=1):
        batch_keys = [k for k in batch_keys if k]
        if not batch_keys:
            continue
        logger.info(
            "Processing batch %d (%d images) from s3://%s/%s ...",
            batch_index,
            len(batch_keys),
            bucket,
            prefix,
        )
        try:
            batch_items = fetch_batch_images(s3, bucket, batch_keys)
            if not batch_items:
                continue
            texts = run_llm_on_batch(llm, sampling_params, batch_items)
            upload_texts_for_keys(
                s3_client=s3,
                bucket=output_bucket,
                output_prefix=output_prefix,
                base_prefix=prefix,
                texts_by_key=texts,
            )
            stats["batches"] += 1
            stats["images"] += len(batch_items)
        except ClientError as exc:
            stats["errors"] += 1
            logger.exception("S3 error while processing batch %d: %s", batch_index, exc)
        except Exception as exc:  # pragma: no cover
            stats["errors"] += 1
            logger.exception("Processing error for batch %d: %s", batch_index, exc)

    summarize(stats)


if __name__ == "__main__":
    main()


