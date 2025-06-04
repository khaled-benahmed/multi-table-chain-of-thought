import os
import json
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login, HfApi
from getpass import getpass

class MultiTableDatasetUploader:
    """
    Class to upload the multi-table WikiTableQuestions dataset to Hugging Face Hub
    """
    
    def __init__(self, dataset_path="wikitablequestions_multi_table", 
                 dataset_name="wikitablequestions-multi-table"):
        """
        Initialize the uploader
        
        Args:
            dataset_path: Path to the directory containing the multi-table dataset files
            dataset_name: Name to use for the Hugging Face dataset
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.api = HfApi()
        
    def load_multi_table_dataset(self):
        """
        Load the multi-table dataset from local files
        
        Returns:
            A DatasetDict containing the dataset splits
        """
        print(f"Loading dataset from {self.dataset_path}...")
        
        dataset_dict = {}
        
        # Load each split
        for split in ['train', 'validation', 'test']:
            file_path = os.path.join(self.dataset_path, f"{split}_multi_table.json")
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {split} split")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to format suitable for datasets library
            processed_data = []
            for item in data:
                # Convert tables to string representation to maintain structure
                tables_json = json.dumps(item['tables'])
                
                processed_data.append({
                    'id': item['id'],
                    'question': item['question'],
                    'answers': item['answers'],
                    'tables_json': tables_json
                })
            
            # Create Dataset object
            dataset_dict[split] = Dataset.from_list(processed_data)
            print(f"Loaded {len(processed_data)} examples for {split} split")
        
        return DatasetDict(dataset_dict)
    
    def authenticate(self, token=None):
        """
        Authenticate with Hugging Face Hub
        
        Args:
            token: Hugging Face token, will prompt if not provided
        """
        if not token:
            print("Please enter your Hugging Face token (or set the HUGGINGFACE_TOKEN env variable):")
            token = getpass("Token: ")
            
        # Login to Hugging Face
        login(token)
        print("Successfully authenticated with Hugging Face")
    
    def upload(self, namespace=None, private=False, token=None):
        """
        Upload the dataset to Hugging Face Hub
        
        Args:
            namespace: User or organization namespace, defaults to authenticated user
            private: Whether the repository should be private
            token: Hugging Face token, will prompt if not provided
        """
        # Authenticate with Hugging Face
        self.authenticate(token)
        
        # Load the dataset
        dataset_dict = self.load_multi_table_dataset()
        
        # Determine the full repository name
        if namespace:
            repo_name = f"{namespace}/{self.dataset_name}"
        else:
            # Use the authenticated user's namespace
            repo_name = f"{self.dataset_name}"
        
        print(f"Uploading dataset to {repo_name}...")
        
        # Push to the Hugging Face Hub
        dataset_dict.push_to_hub(
            repo_name,
            private=private
        )
        
        print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")
        print("\nUsage example:")
        print(f"""
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{repo_name}")

# Access an example
example = dataset["train"][0]

# Convert tables_json back to Python structure
tables = json.loads(example["tables_json"])

# Now you can use the tables in your application
print(f"Question: {{example['question']}}")
print(f"Number of tables: {{len(tables)}}")
print(f"Table names: {{[table['name'] for table in tables]}}")
""")

def main():
    """Command line interface for the uploader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload multi-table WikiTableQuestions dataset to Hugging Face Hub')
    parser.add_argument('--path', default="wikitablequestions_multi_table", 
                        help='Path to the multi-table dataset directory')
    parser.add_argument('--name', default="wikitablequestions-multi-table",
                        help='Name for the dataset on Hugging Face Hub')
    parser.add_argument('--namespace', default=None,
                        help='User or organization namespace (defaults to authenticated user)')
    parser.add_argument('--private', action='store_true',
                        help='Make the repository private')
    parser.add_argument('--token', default=os.environ.get('HUGGINGFACE_TOKEN'),
                        help='Hugging Face token (will prompt if not provided)')
    
    args = parser.parse_args()
    
    uploader = MultiTableDatasetUploader(args.path, args.name)
    uploader.upload(args.namespace, args.private, args.token)

if __name__ == "__main__":
    main()