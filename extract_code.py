import json

def extract():
    with open(r'c:\ProgFiles\IntelligentFraudDetection\notebook8d8ac1fa3a.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(r'c:\ProgFiles\IntelligentFraudDetection\project.py', 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                # Only write out if it's not empty and not a pip install command
                if source.strip() and not source.strip().startswith('!'):
                    f.write(source + '\n\n')

if __name__ == '__main__':
    extract()
