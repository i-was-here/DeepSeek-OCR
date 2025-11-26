#!/usr/bin/env python3
"""
Convert every PDF under an S3 prefix into JPG images.

Usage:
    ./pdf-to-imgs my-bucket path/to/pdfs

Environment variables:
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION  # standard AWS auth
    PDF_TO_IMG_DPI (default: 144)                           # rendering DPI
    PDF_IMG_UPLOAD_WORKERS (default: 8)                     # per-PDF upload pool
    PDF_IMG_OUTPUT_PREFIX (default: images)                 # root folder for images
    PDF_IMG_OUTPUT_BUCKET (default: same as source bucket)  # override destination
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple
from dotenv import load_dotenv
from tqdm.auto import tqdm

import boto3
from botocore.exceptions import ClientError

try:
    import fitz  # type: ignore  # PyMuPDF
except ImportError as exc:  # pragma: no cover - configuration error
    raise ImportError("PyMuPDF (install via `pip install pymupdf`) is required.") from exc

load_dotenv()

DEFAULT_DPI = int(os.getenv("PDF_TO_IMG_DPI", "144"))
UPLOAD_WORKERS = int(os.getenv("PDF_IMG_UPLOAD_WORKERS", "8"))
OUTPUT_PREFIX = os.getenv("PDF_IMG_OUTPUT_PREFIX", "images").strip("/")
OUTPUT_BUCKET = os.getenv("PDF_IMG_OUTPUT_BUCKET")


@dataclass
class PdfResult:
    key: str
    pages: int
    uploaded: int
    failed_uploads: int
    error: str | None = None


def _build_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


def iter_pdf_keys(s3_client, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if key.lower().endswith(".pdf"):
                yield key


def download_pdf(s3_client, bucket: str, prefix: str, key: str) -> bytes | None:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=f"{prefix}/{key}")
        return response["Body"].read()
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code == "NoSuchKey":
            logging.warning("Missing PDF %s/%s", bucket, key)
            return None
        logging.error("Failed to download s3://%s/%s: %s", bucket, key, exc)
        return None


def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = DEFAULT_DPI) -> List[Tuple[int, bytes]]:
    images: List[Tuple[int, bytes]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in range(doc.page_count):
            pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
            images.append((idx + 1, pix.tobytes("jpeg", jpg_quality=95)))
    finally:
        doc.close()
    return images


def _upload_single_image(args: Tuple) -> Tuple[int, bool, str | None]:
    s3_client, bucket, prefix, key, page_num, image_bytes = args
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
        return page_num, True, None
    except Exception as exc:  # pragma: no cover - best effort logging
        return page_num, False, str(exc)


def _destination_key(pdf_key: str, page_num: int) -> str:
    stripped = pdf_key[:-4] if pdf_key.lower().endswith(".pdf") else pdf_key
    stripped = stripped.strip("/")
    return f"{OUTPUT_PREFIX}/{stripped}/page_{page_num}.jpg"


def _pdf_output_exists(s3_client, bucket: str, pdf_key: str) -> bool:
    """
    Check if the output folder for this PDF already exists in the destination bucket.

    We consider the folder to exist if there is at least one object under
    OUTPUT_PREFIX/{pdf_key_without_ext}/.
    """
    stripped = pdf_key[:-4] if pdf_key.lower().endswith(".pdf") else pdf_key
    stripped = stripped.strip("/")
    prefix = f"{OUTPUT_PREFIX}/{stripped}/"

    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def upload_images(
    s3_client,
    images: Sequence[Tuple[int, bytes]],
    destination_bucket: str,
    pdf_key: str,
) -> Tuple[int, int]:
    uploaded = 0
    failures = 0
    if not images:
        return uploaded, failures

    if len(images) == 1 or UPLOAD_WORKERS <= 1:
        for page, payload in images:
            key = _destination_key(pdf_key, page)
            _, ok, _ = _upload_single_image((s3_client, destination_bucket, OUTPUT_PREFIX, key, page, payload))
            if ok:
                uploaded += 1
            else:
                failures += 1
        return uploaded, failures

    with ThreadPoolExecutor(max_workers=min(len(images), UPLOAD_WORKERS)) as executor:
        tasks = []
        for page, payload in images:
            key = _destination_key(pdf_key, page)
            tasks.append(
                executor.submit(
                    _upload_single_image, (_build_s3_client(), destination_bucket, OUTPUT_PREFIX, key, page, payload)
                )
            )
        for future in as_completed(tasks):
            page_num, ok, error = future.result()
            if ok:
                uploaded += 1
            else:
                failures += 1
                logging.error(
                    "Failed to upload %s page %d: %s", pdf_key, page_num, error or "unknown error"
                )
    return uploaded, failures


def process_pdf(s3_client, bucket: str, prefix: str, pdf_key: str) -> PdfResult:
    dest_bucket = OUTPUT_BUCKET or bucket
    # Skip processing if output folder already exists
    try:
        if _pdf_output_exists(s3_client, dest_bucket, pdf_key):
            logging.info(
                "Skipping %s because output folder already exists in s3://%s",
                pdf_key,
                dest_bucket,
            )
            return PdfResult(
                key=pdf_key,
                pages=0,
                uploaded=0,
                failed_uploads=0,
                error="skipped_existing_output",
            )
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.warning(
            "Failed to check existing output for %s in bucket %s: %s",
            pdf_key,
            dest_bucket,
            exc,
        )

    pdf_bytes = download_pdf(s3_client, bucket, prefix, pdf_key)
    if pdf_bytes is None:
        return PdfResult(key=pdf_key, pages=0, uploaded=0, failed_uploads=0, error="download_failed")

    try:
        images = convert_pdf_to_images(pdf_bytes)
    except Exception as exc:
        logging.error("Conversion failed for %s: %s", pdf_key, exc)
        return PdfResult(key=pdf_key, pages=0, uploaded=0, failed_uploads=0, error="conversion_failed")

    uploaded, failed = upload_images(s3_client, images, dest_bucket, pdf_key)
    error = None
    if failed:
        error = f"{failed} uploads failed"
    return PdfResult(
        key=pdf_key,
        pages=len(images),
        uploaded=uploaded,
        failed_uploads=failed,
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDFs under an S3 prefix into images.")
    parser.add_argument("bucket", help="S3 bucket containing the PDF files")
    parser.add_argument("--prefix", help="Prefix inside the bucket to scan for PDFs", default="pdfs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s3_client = _build_s3_client()
    pdf_keys = list(iter_pdf_keys(s3_client, args.bucket, args.prefix))
    # remove the prefix from the pdf_keys
    pdf_keys = [key.replace(args.prefix + "/", "") for key in pdf_keys]
    if not pdf_keys:
        logging.info("No PDF objects found under s3://%s/%s", args.bucket, args.prefix)
        return

    logging.info("Found %d PDF(s) to process under s3://%s/%s", len(pdf_keys), args.bucket, args.prefix)
    results: List[PdfResult] = []
    for key in tqdm(pdf_keys):
        logging.info("Processing %s ...", key)
        result = process_pdf(s3_client, args.bucket, args.prefix, key)
        results.append(result)
        if result.error:
            logging.warning(
                "Completed %s with %d/%d uploads (error=%s)",
                key,
                result.uploaded,
                result.pages,
                result.error,
            )
        else:
            logging.info("Completed %s with %d pages uploaded", key, result.uploaded)

    total = len(results)
    uploaded_pages = sum(r.uploaded for r in results)
    failed_pages = sum(r.failed_uploads for r in results)
    failed_docs = [r for r in results if r.error]

    logging.info("===== Summary =====")
    logging.info("PDFs processed: %d", total)
    logging.info("Pages uploaded: %d", uploaded_pages)
    logging.info("Failed uploads: %d", failed_pages)
    if failed_docs:
        logging.warning("Documents with issues: %d", len(failed_docs))
        for doc in failed_docs[:10]:
            logging.warning(" - %s (%s)", doc.key, doc.error)
        if len(failed_docs) > 10:
            logging.warning(" - ... %d more", len(failed_docs) - 10)
    logging.info("All done.")


if __name__ == "__main__":
    main()

