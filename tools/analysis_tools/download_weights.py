import os
import yaml
import glob
import subprocess

def print_summary(downloaded_items, skipped_items, failed_items):
    """Print execution summary, listing downloaded, skipped, and failed items separately."""
    print('\n\n' + '=' * 80)
    print('DOWNLOAD SUMMARY')
    print('=' * 80 + '\n')

    # Downloaded items block
    if downloaded_items:
        print(f'Successfully downloaded: {len(downloaded_items)} item(s)')
        for idx, item in enumerate(downloaded_items, 1):
            print(f'    {idx}. {item["filename"]}')
        print()

    # Skipped items block
    if skipped_items:
        print(f'Already downloaded and skipped: {len(skipped_items)} item(s)\n')

    # Failed items block
    if not failed_items:
        print('Status: All required files were processed successfully!')
    else:
        print(f'Status: Failed to download {len(failed_items)} item(s) due to network or server errors:\n')
        for idx, item in enumerate(failed_items, 1):
            print(f'    {idx}. {item["name"]}')
            print(f'       Filename:    {item["filename"]}')
            print(f'       Wget Code:   {item.get("error_code", "Unknown")}')
            print(f'       Manual Link: {item["url"]}\n')
            
        print('    Tip: You can manually download these files via browser using the links above,')
        print('    and place them in the checkpoints/ folder.')
        
    print('\n' + '=' * 80 + '\n')


def main():
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metafiles = glob.glob('configs/**/metafile.yml', recursive=True)
    
    if not metafiles:
        print("No metafile.yml files found in configs/.")
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
                    print(f"Skipping:  {filename}")
                    skipped_items.append(item_info)
                    continue
                
                print(f"Downloading: {filename}")
                
                try:
                    result = subprocess.run(
                        ['wget', '-q', '--show-progress', weights_url, '-O', save_path]
                    )
                    
                    if result.returncode == 0:
                        downloaded_items.append(item_info)
                    else:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                            print(f"\nRemoved invalid/HTML file created by failed download: {filename}\n")
                        
                        item_info['error_code'] = result.returncode
                        failed_items.append(item_info)
                except Exception as e:
                    print(f"Error while running wget: {e}")
                    item_info['error_code'] = 'Exception'
                    failed_items.append(item_info)
                    
    print_summary(downloaded_items, skipped_items, failed_items)

if __name__ == '__main__':
    main()