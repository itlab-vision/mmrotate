import argparse
import logging
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger("dota_downloader")

# Configuration & Datasets
# Default root directory for dataset storage
DATA_ROOT_DIR = Path('./data')

# Format: (split_type, target_relative_path, display_name, google_drive_id)

DOTA_1_0_ITEMS = [
    # Training set
    ('train', 'train/images', 'train/images part_1', '1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2'),
    ('train', 'train/images', 'train/images part_2', '1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v'),
    ('train', 'train/images', 'train/images part_3', '1pEmwJtugIWhiwgBqOtplNUtTG2T454zn'),
    ('train', 'train/labelTxt', 'train/labelTxt', '1I-faCP-DOxf6mxcjUTc8mYVPqUgSQxx6'),
    ('train', 'train/labelTxtHbb', 'train/labelTxtHbb', '1sS9hveKtYAiTsGVxC4msF5qJjhn3wYpY'),

    # Validation set
    ('val', 'val/images', 'val/images part_1', '1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP'),
    ('val', 'val/labelTxt', 'val/labelTxt', '1uFwxA4B7H8zcI1oD11bj0U8z88qroMlG'),
    ('val', 'val/labelTxtHbb', 'val/labelTxtHbb', '1roMkDBK9753uS5tCmtYlRTyzrObjjJ83'),

    # Testing set
    ('test', 'test/images', 'test/images part_1', '1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK'),
    ('test', 'test/images', 'test/images part_2', '1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv'),
]

DOTA_1_5_ITEMS = [
    # Training set
    ('train', 'train/images', 'train/images part_1', '1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2'),
    ('train', 'train/images', 'train/images part_2', '1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v'),
    ('train', 'train/images', 'train/images part_3', '1pEmwJtugIWhiwgBqOtplNUtTG2T454zn'),
    ('train', 'train/labelTxt', 'train/labelTxt', '12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX'),
    ('train', 'train/labelTxtHbb', 'train/labelTxtHbb', '1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb'),

    # Validation set
    ('val', 'val/images', 'val/images part_1', '1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP'),
    ('val', 'val/labelTxt', 'val/labelTxt', '1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC'),
    ('val', 'val/labelTxtHbb', 'val/labelTxtHbb', '1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse'),

    # Testing set
    ('test', 'test/images', 'test/images part_1', '1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK'),
    ('test', 'test/images', 'test/images part_2', '1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv'),
]

DOTA_2_0_ITEMS = [
    # Add entries when links are available
]

TEMP_DIR = Path('./.dota_tmp_download')


def init_logger():
    """Initializes global logger for clean console output."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='Download and prepare DOTA dataset')
    parser.add_argument(
        '--version',
        type=str,
        nargs='+',
        choices=['1.0', '1.5', '2.0', 'all'],
        default=['1.0'],
        help='version(s) of DOTA dataset to download (default: "1.0")')
    parser.add_argument(
        '--split',
        type=str,
        nargs='+',
        choices=['train', 'val', 'test', 'all'],
        default=['val'],
        help='dataset split(s) to download (default: "val")')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=str(DATA_ROOT_DIR),
        help=f'directory where dataset will be saved (default: "{DATA_ROOT_DIR}")')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='force re-download and overwrite existing data')

    args = parser.parse_args()
    return args


def check_gdown():
    """Ensure gdown library is installed."""
    try:
        import gdown
        return gdown
    except ImportError:
        logger.error('ERROR: "gdown" package is missing. Please install it via: pip install gdown')
        sys.exit(1)


def load_manifest(manifest_path: Path) -> set:
    """Load set of already processed entries from manifest file."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def mark_as_downloaded(manifest_path: Path, manifest_key: str):
    """Append a unique manifest key to the manifest file after successful extraction."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'a', encoding='utf-8') as f:
        f.write(f'{manifest_key}\n')


def download_file(gdown_module, file_id: str, output_path: Path):
    """Download a file from Google Drive using gdown API."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown_module.download(url, str(output_path), quiet=False)


def extract_and_place(archive_path: Path, target_dir: Path):
    """Extract .zip or .tar archive and move inner files to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_tmp = TEMP_DIR / 'extract_tmp'

    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    logger.info(f'    Extracting {archive_path.name}...')

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_tmp)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.TarFile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_tmp)
    else:
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_tmp)
        except Exception as e:
            logger.error(f'ERROR: Failed to extract {archive_path}: {e}')
            return

    # Determine source directory
    # If archive contains a single top-level directory (e.g. 'images/' or 'labelTxtHbb/'), navigate into it
    top_level_contents = list(extract_tmp.iterdir())
    if len(top_level_contents) == 1 and top_level_contents[0].is_dir():
        source_dir = top_level_contents[0]
    else:
        source_dir = extract_tmp

    # Copy/move files into target directory
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    shutil.rmtree(extract_tmp)


def process_version(gdown_module, version_name: str, base_dir: Path, items: list, selected_splits: set, manifest_path: Path, overwrite: bool = False):
    """Process downloading and extraction for a specific dataset version."""
    if not items:
        logger.info(f'\nSkipping {version_name} (no links provided).')
        return [], []

    logger.info(f'\nProcessing {version_name} -> {base_dir}')

    processed_keys = load_manifest(manifest_path)
    filtered_items = [item for item in items if item[0] in selected_splits]
    total_items = len(filtered_items)

    failed_items = []
    skipped_items = []

    for idx, (split_type, rel_target, display_name, file_id) in enumerate(filtered_items, 1):
        # Skip items without Google Drive ID
        if not file_id:
            logger.info(f'\n[{idx}/{total_items}] [{version_name}] "{display_name}" has no Google Drive ID provided. Skipping.')
            continue

        full_target_dir = base_dir / rel_target
        manifest_key = f'{version_name}|{rel_target}|{file_id}'

        if not overwrite and manifest_key in processed_keys:
            logger.info(f'\n[{idx}/{total_items}] [{version_name}] "{display_name}" already processed. Skipping.')
            skipped_items.append({
                'version': version_name,
                'display_name': display_name
            })
            continue

        temp_archive = TEMP_DIR / f'{version_name}_part_{idx}.tmp'

        logger.info(f'\n[{idx}/{total_items}] Downloading {version_name} ({display_name})...')

        try:
            download_file(gdown_module, file_id, temp_archive)
            extract_and_place(temp_archive, full_target_dir)
            mark_as_downloaded(manifest_path, manifest_key)
        except Exception:
            logger.error(f'WARNING: Failed to download {version_name} ({display_name}). Skipped.')
            failed_items.append({
                'version': version_name,
                'display_name': display_name,
                'file_id': file_id,
                'url': f'https://drive.google.com/file/d/{file_id}/view'
            })
        finally:
            if temp_archive.exists():
                temp_archive.unlink()

    return failed_items, skipped_items


def print_summary(failed_items, skipped_items):
    """Print execution summary, listing skipped and failed items separately."""
    logger.info('\n' + '=' * 60)
    logger.info('DOWNLOAD SUMMARY')
    logger.info('=' * 60)

    if skipped_items:
        logger.info(f'Already downloaded and skipped ({len(skipped_items)} item(s)):')
        for idx, item in enumerate(skipped_items, 1):
            logger.info(f'  {idx}. [{item["version"]}] {item["display_name"]}')
        logger.info('')

    if not failed_items:
        logger.info('Status: All required files were processed successfully!')
    else:
        logger.error(f'Status: Failed to download {len(failed_items)} item(s) due to Google Drive quotas or errors:\n')
        for idx, item in enumerate(failed_items, 1):
            logger.error(f'  {idx}. [{item["version"]}] {item["display_name"]}')
            logger.error(f'     Google Drive ID: {item["file_id"]}')
            logger.error(f'     Manual Link:     {item["url"]}\n')
        logger.info('Tip: You can manually download these files via browser using the links above,')
        logger.info('and place their contents in appropriate folders.')
    logger.info('=' * 60)


def main():
    init_logger()
    args = parse_args()

    selected_versions = set(args.version)
    if 'all' in selected_versions:
        selected_versions = {'1.0', '1.5', '2.0'}

    selected_splits = set(args.split)
    if 'all' in selected_splits:
        selected_splits = {'train', 'val', 'test'}

    out_base = Path(args.out_dir)
    manifest_path = out_base / '.downloaded_manifest.txt'

    gdown_module = check_gdown()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    all_failed_items = []
    all_skipped_items = []

    try:
        if '1.0' in selected_versions:
            failed, skipped = process_version(
                gdown_module=gdown_module,
                version_name='DOTA-v1.0',
                base_dir=out_base / 'DOTA_1_0',
                items=DOTA_1_0_ITEMS,
                selected_splits=selected_splits,
                manifest_path=manifest_path,
                overwrite=args.overwrite
            )
            all_failed_items.extend(failed)
            all_skipped_items.extend(skipped)

        if '1.5' in selected_versions:
            failed, skipped = process_version(
                gdown_module=gdown_module,
                version_name='DOTA-v1.5',
                base_dir=out_base / 'DOTA_1_5',
                items=DOTA_1_5_ITEMS,
                selected_splits=selected_splits,
                manifest_path=manifest_path,
                overwrite=args.overwrite
            )
            all_failed_items.extend(failed)
            all_skipped_items.extend(skipped)

        if '2.0' in selected_versions:
            failed, skipped = process_version(
                gdown_module=gdown_module,
                version_name='DOTA-v2.0',
                base_dir=out_base / 'DOTA_2_0',
                items=DOTA_2_0_ITEMS,
                selected_splits=selected_splits,
                manifest_path=manifest_path,
                overwrite=args.overwrite
            )
            all_failed_items.extend(failed)
            all_skipped_items.extend(skipped)
    finally:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    print_summary(all_failed_items, all_skipped_items)


if __name__ == '__main__':
    main()