import os
import sys
import re
import shutil
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Iterable

import psycopg2
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

# Ensure DeepSeek-OCR-vllm package modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VLLM_DIR = os.path.join(PROJECT_ROOT, 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm')
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)

# DeepSeek OCR / vLLM imports
import torch  # noqa: F401 - required to set env and drive GPU
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, PROMPT, CROP_MODE, MAX_CONCURRENCY


def setup_logger(verbose: bool):
    logger = logging.getLogger('img-to-txt')
    if verbose:
        # Configure only when verbose; otherwise keep silent
        level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        )
        logger.disabled = False
    else:
        # Disable this logger entirely when not verbose
        logger.disabled = True
    return logger


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    # Accept formats like: s3://bucket, s3://bucket/, bucket, bucket/
    s = s3_url.strip()
    s = s.replace('s3://', '').replace('S3://', '')
    s = s.split(':', 1)[-1] if ':' in s else s
    s = s.lstrip('/')
    if '/' in s:
        bucket, prefix = s.split('/', 1)
    else:
        bucket, prefix = s, ''
    prefix = prefix.strip('/')
    return bucket, prefix


def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def db_connect():
    load_dotenv()
    logging.getLogger('img-to-txt').info('Connecting to PostgreSQL...')
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME', 'postgres'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD'),
        sslmode=os.getenv('DB_SSLMODE', 'require'),
    )
    logging.getLogger('img-to-txt').info('Connected to PostgreSQL.')
    return conn


def fetch_pending_media(conn) -> List[Tuple[str, str]]:
    # returns list of (media_id, document_type)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT media_id, document_type
            FROM table_3
            WHERE status = 2 AND document_type IN ('pdf','jpg','jpeg','png')
            """
        )
        rows = cur.fetchall()
    result = [(str(r[0]), str(r[1]).lower()) for r in rows]
    logging.getLogger('img-to-txt').info(f'Fetched {len(result)} pending media rows (status=2).')
    return result


def update_status_done(conn, media_id: str):
    with conn.cursor() as cur:
        cur.execute("UPDATE table_3 SET status = 3 WHERE media_id = %s", (media_id,))
    conn.commit()
    logging.getLogger('img-to-txt').info(f'Updated media_id={media_id} to status=3.')


def s3_client():
    logging.getLogger('img-to-txt').info('Creating S3 client...')
    max_pool = env_int('AWS_MAX_POOL_CONNECTIONS', 64)
    client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION'),
        config=Config(
            max_pool_connections=max_pool,
            retries={'max_attempts': env_int('AWS_MAX_RETRIES', 5), 'mode': 'standard'},
            tcp_keepalive=True,
        )
    )
    logging.getLogger('img-to-txt').info('S3 client ready.')
    return client


def list_media_pages(s3, bucket: str, base_prefix: str, media_id: str) -> List[str]:
    # Pages live under: <base_prefix>/images/<media_id>/
    prefix = '/'.join([p for p in [base_prefix, 'images', media_id] if p]) + '/'
    paginator = s3.get_paginator('list_objects_v2')
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get('Contents', [])
        for obj in contents:
            key = obj['Key']
            base = os.path.basename(key)
            if re.match(r'^page_\d+\.(jpg|jpeg|png)$', base, flags=re.IGNORECASE):
                keys.append(key)
    keys.sort()
    logging.getLogger('img-to-txt').info(f'media_id={media_id}: found {len(keys)} page images at s3://{bucket}/{prefix}')
    return keys


def download_objects(s3, bucket: str, keys: List[str], dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    logging.getLogger('img-to-txt').info(f'Downloading {len(keys)} objects to {dest_dir} ...')
    def _dl(key: str):
        local_path = os.path.join(dest_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        return local_path
    max_workers = min(len(keys) or 1, env_int('S3_DOWNLOAD_WORKERS', 16))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_dl, keys), total=len(keys), desc=f"Downloading -> {dest_dir}", leave=False))
    logging.getLogger('img-to-txt').info(f'Download complete: {dest_dir}')


def build_llm_and_params():
    logging.getLogger('img-to-txt').info('Initializing DeepSeek OCR model & vLLM ...')
    if torch.version.cuda == '11.8':
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")
    os.environ.setdefault('VLLM_USE_V1', '0')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

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
        gpu_memory_utilization=0.9,
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
    logging.getLogger('img-to-txt').info('Model ready.')
    return llm, sampling_params


def build_batch_inputs(image_paths: List[str]) -> List[Dict]:
    processor = DeepseekOCRProcessor()
    inputs = []
    for p in image_paths:
        image = Image.open(p).convert('RGB')
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
    formula_pattern = r'\\\[(.*?)\\\]'
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        return r'\\[' + formula.strip() + r'\\]'
    return re.sub(formula_pattern, process_formula, text, flags=re.DOTALL)


def strip_ref_det_blocks(text: str) -> str:
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    for full, _, _ in matches:
        text = text.replace(full, '')
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
    return text


def upload_markdowns(s3, bucket: str, base_prefix: str, media_id: str, outputs: Dict[str, str]):
    # outputs: mapping local_image_path -> markdown_content
    def _put(item: Tuple[str, str]):
        local_path, content = item
        base = os.path.basename(local_path)
        page_stem = os.path.splitext(base)[0]
        key = '/'.join([p for p in [base_prefix, 'markdowns', media_id, f'{page_stem}.md'] if p])
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'), ContentType='text/markdown')

    items = list(outputs.items())
    max_workers = min(len(items) or 1, env_int('S3_UPLOAD_WORKERS', 16))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_put, items))
    logging.getLogger('img-to-txt').info(f'media_id={media_id}: uploaded {len(items)} markdown files to s3://{bucket}/{base_prefix or ""}')


def process_media_folder(
    llm: LLM,
    sampling_params: SamplingParams,
    s3,
    bucket: str,
    base_prefix: str,
    media_id: str,
    mini_batch_size: int,
    download_root: str,
):
    # 1) List & download
    keys = list_media_pages(s3, bucket, base_prefix, media_id)
    if not keys:
        logging.getLogger('img-to-txt').warning(f'media_id={media_id}: no page images found; skipping.')
        return False  # nothing to do

    local_dir = os.path.join(download_root, media_id)
    download_objects(s3, bucket, keys, local_dir)

    # 2) Iterate pages in mini-batches, run OCR, upload
    page_paths = sorted([
        os.path.join(local_dir, f)
        for f in os.listdir(local_dir)
        if re.match(r'^page_\d+\.(jpg|jpeg|png)$', f, flags=re.IGNORECASE)
    ])

    logger = logging.getLogger('img-to-txt')
    for batch in tqdm(list(chunked(page_paths, mini_batch_size)), desc=f"OCR {media_id}", leave=False):
        batch_inputs = build_batch_inputs(batch)
        logger.info(f'media_id={media_id}: running OCR for {len(batch)} pages ...')
        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

        results: Dict[str, str] = {}
        for output, img_path in zip(outputs_list, batch):
            content = output.outputs[0].text
            content = clean_formula(content)
            content = strip_ref_det_blocks(content)
            results[img_path] = content

        upload_markdowns(s3, bucket, base_prefix, media_id, results)
        logger.info(f'media_id={media_id}: batch processed and uploaded ({len(results)} pages).')

    # 3) Cleanup disk
    shutil.rmtree(local_dir, ignore_errors=True)
    logger.info(f'media_id={media_id}: cleaned up local folder {local_dir}.')
    return True


def main():
    # Parse flags early to control logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-batch-size', type=int, default=env_int('FOLDER_BATCH_SIZE', 40))
    parser.add_argument('--mini-page-batch-size', type=int, default=env_int('MINI_PAGE_BATCH_SIZE', 20))
    parser.add_argument('--download-tmp-dir', type=str, default=os.getenv('DOWNLOAD_TMP_DIR', '/tmp/ds_ocr_downloads'))
    parser.add_argument('--s3-url', type=str, default=os.getenv('S3_URL', 's3://'))
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging', default=False)
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    bucket, base_prefix = parse_s3_url(args.s3_url)
    if not bucket:
        raise ValueError('S3_URL must specify a bucket, e.g., s3://my-bucket')

    os.makedirs(args.download_tmp_dir, exist_ok=True)
    logger.info(f'Run config: bucket={bucket}, prefix={base_prefix}, folder_batch_size={args.folder_batch_size}, mini_page_batch_size={args.mini_page_batch_size}, tmp_dir={args.download_tmp_dir}')

    # Initialize once
    llm, sampling_params = build_llm_and_params()
    s3 = s3_client()

    # DB connection
    conn = db_connect()
    try:
        pending = fetch_pending_media(conn)
        if not pending:
            logger.info('No pending media to process (status=2).')
            return

        media_ids = [m for m, _ in pending]

        for folder_batch in tqdm(list(chunked(media_ids, args.folder_batch_size)), desc='Folder batches'):
            logger.info(f'Processing folder batch of size {len(folder_batch)} ...')
            for media_id in folder_batch:
                processed = False
                try:
                    logger.info(f'Start processing media_id={media_id} ...')
                    processed = process_media_folder(
                        llm=llm,
                        sampling_params=sampling_params,
                        s3=s3,
                        bucket=bucket,
                        base_prefix=base_prefix,
                        media_id=media_id,
                        mini_batch_size=args.mini_page_batch_size,
                        download_root=args.download_tmp_dir,
                    )
                except ClientError as ce:
                    logger.exception(f"S3 error for media_id={media_id}: {ce}")
                except Exception as e:
                    logger.exception(f"Processing error for media_id={media_id}: {e}")
                finally:
                    # Free any leftover folder in case of partial failure
                    shutil.rmtree(os.path.join(args.download_tmp_dir, media_id), ignore_errors=True)

                if processed:
                    try:
                        update_status_done(conn, media_id)
                    except Exception as e:
                        logger.exception(f"Failed to update DB status for media_id={media_id}: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()


