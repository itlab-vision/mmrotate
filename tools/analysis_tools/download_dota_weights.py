import os
import sys
import yaml
import glob
import logging
import subprocess

logger = logging.getLogger("downloader")


def init_logger():
    """Initializes global logger for clean console output."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)


def print_summary(downloaded_items, skipped_items, failed_items):
    """Prints execution summary, listing downloaded, skipped, and failed items separately."""
    logger.info('\n\n' + '=' * 80)
    logger.info('DOWNLOAD SUMMARY')
    logger.info('=' * 80 + '\n')

    # Downloaded items block
    if downloaded_items:
        logger.info(f'Successfully downloaded: {len(downloaded_items)} item(s)')
        for idx, item in enumerate(downloaded_items, 1):
            logger.info(f'    {idx}. {item["filename"]}')
        logger.info('')

    # Skipped items block
    if skipped_items:
        logger.info(f'Already downloaded and skipped: {len(skipped_items)} item(s)\n')

    # Failed items block
    if not failed_items:
        logger.info('Status: All required files were processed successfully!')
    else:
        logger.error(f'Status: Failed to download {len(failed_items)} item(s) due to network or server errors:\n')
        for idx, item in enumerate(failed_items, 1):
            logger.error(f'    {idx}. {item["name"]}')
            logger.error(f'       Filename:    {item["filename"]}')
            logger.error(f'       Wget Code:   {item.get("error_code", "Unknown")}')
            logger.error(f'       Manual Link: {item["url"]}\n')
            
        logger.info('    Tip: You can manually download these files via browser using the links above,')
        logger.info('    and place them in the checkpoints/ folder.')
        
    logger.info('\n' + '=' * 80 + '\n')


def main():
    init_logger()

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metafiles = glob.glob('configs/**/metafile.yml', recursive=True)
    
    if not metafiles:
        logger.error("No metafile.yml files found in configs/.")
        return

    downloaded_items = []
    skipped_items = []
    failed_items = []
    
    for mf_path in metafiles:
        with open(mf_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                continue
            
        if not data or 'Models' not in data:
            continue
            
        for model in data['Models']:
            name = model.get('Name', 'Unknown Model')
            config = model.get('Config', '')
            weights_url = model.get('Weights', '')
            meta = model.get('Metadata', {})
            training_data = meta.get('Training Data', '').lower()

            if 'dota' not in training_data and 'dota' not in config.lower():
                continue
            
            if weights_url.startswith('http'):
                filename = os.path.basename(weights_url)
                save_path = os.path.join(checkpoint_dir, filename)
                
                item_info = {
                    'name': name,
                    'url': weights_url,
                    'filename': filename
                }
                
                if os.path.exists(save_path):
                    logger.info(f"Skipping:  {filename}")
                    skipped_items.append(item_info)
                    continue
                
                logger.info(f"Downloading: {filename}")
                
                try:
                    result = subprocess.run(
                        ['wget', '-q', '--show-progress', weights_url, '-O', save_path]
                    )
                    
                    if result.returncode == 0:
                        downloaded_items.append(item_info)
                    else:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                            logger.error(f"\nRemoved invalid/HTML file created by failed download: {filename}\n")
                        
                        item_info['error_code'] = result.returncode
                        failed_items.append(item_info)
                except Exception as e:
                    logger.error(f"Error while running wget: {e}")
                    item_info['error_code'] = 'Exception'
                    failed_items.append(item_info)
                    
    print_summary(downloaded_items, skipped_items, failed_items)

if __name__ == '__main__':
    main()