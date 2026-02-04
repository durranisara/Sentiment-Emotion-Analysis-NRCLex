"""
Data Loader Module
==================

Handles loading and preprocessing of various data formats for sentiment analysis.
Supports CSV, JSON, TXT files, and direct data scraping utilities.

Author: Muhammad Awais
Email: mawaiskhan1808@gmail.com
"""

import pandas as pd
import json
import csv
from pathlib import Path
from typing import List, Dict, Union, Optional, Generator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for sentiment analysis projects.
    
    Supports multiple formats and provides preprocessing options.
    """
    
    SUPPORTED_FORMATS = ['.csv', '.json', '.jsonl', '.txt']
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            file_path (str, optional): Path to data file
        """
        self.file_path = file_path
        self.data = None
        
        if file_path:
            self.file_path = Path(file_path)
            self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that file exists and format is supported."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.file_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
    
    def load_csv(
        self, 
        text_column: str = 'text',
        label_column: Optional[str] = None,
        encoding: str = 'utf-8',
        **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            text_column (str): Name of column containing text data
            label_column (str, optional): Name of column containing labels
            encoding (str): File encoding
            **pandas_kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading CSV: {self.file_path}")
        
        try:
            df = pd.read_csv(self.file_path, encoding=encoding, **pandas_kwargs)
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, trying latin-1 encoding")
            df = pd.read_csv(self.file_path, encoding='latin-1', **pandas_kwargs)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found. "
                           f"Available: {list(df.columns)}")
        
        # Select relevant columns
        columns = [text_column]
        if label_column and label_column in df.columns:
            columns.append(label_column)
        
        self.data = df[columns].copy()
        self.data = self.data.dropna(subset=[text_column])
        
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_json(
        self,
        text_key: str = 'text',
        label_key: Optional[str] = None,
        lines: bool = False
    ) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            text_key (str): Key containing text data
            label_key (str, optional): Key containing labels
            lines (bool): Whether file is JSON Lines format
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading JSON: {self.file_path}")
        
        if lines:
            records = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        
        if text_key not in df.columns:
            raise ValueError(f"Text key '{text_key}' not found")
        
        columns = [text_key]
        if label_key and label_key in df.columns:
            columns.append(label_key)
        
        self.data = df[columns].copy()
        self.data = self.data.dropna(subset=[text_key])
        
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_txt(self, delimiter: str = '\n') -> pd.DataFrame:
        """
        Load plain text file.
        
        Args:
            delimiter (str): Delimiter between texts (default: newline)
            
        Returns:
            pd.DataFrame: DataFrame with 'text' column
        """
        logger.info(f"Loading TXT: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        texts = [t.strip() for t in content.split(delimiter) if t.strip()]
        self.data = pd.DataFrame({'text': texts})
        
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Auto-detect and load file based on extension.
        
        Args:
            **kwargs: Format-specific arguments
            
        Returns:
            pd.DataFrame: Loaded data
        """
        ext = self.file_path.suffix.lower()
        
        if ext == '.csv':
            return self.load_csv(**kwargs)
        elif ext in ['.json', '.jsonl']:
            return self.load_json(lines=(ext == '.jsonl'), **kwargs)
        elif ext == '.txt':
            return self.load_txt(**kwargs)
        else:
            raise ValueError(f"Cannot auto-load format: {ext}")
    
    def get_texts(self) -> List[str]:
        """Get list of text strings from loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        text_col = 'text' if 'text' in self.data.columns else self.data.columns[0]
        return self.data[text_col].tolist()
    
    @staticmethod
    def save_results(
        results: List[Dict],
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """
        Save analysis results to file.
        
        Args:
            results (List[Dict]): Analysis results
            output_path (str): Output file path
            format (str): Output format ('csv', 'json', 'jsonl')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Results saved to: {output_path}")


class ReviewDataset:
    """
    Specialized loader for consumer review datasets.
    
    Handles common review formats (Amazon, Yelp, etc.)
    """
    
    @staticmethod
    def load_amazon_reviews(file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load Amazon reviews dataset.
        
        Expected columns: reviewText, overall (rating)
        """
        loader = DataLoader(file_path)
        df = loader.load_csv(text_column='reviewText', label_column='overall')
        
        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)
        
        # Convert rating to sentiment
        df['sentiment_label'] = df['overall'].apply(
            lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
        )
        
        return df
    
    @staticmethod
    def load_yelp_reviews(file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load Yelp reviews dataset.
        
        Expected columns: text, stars
        """
        loader = DataLoader(file_path)
        df = loader.load_csv(text_column='text', label_column='stars')
        
        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)
        
        # Convert stars to sentiment
        df['sentiment_label'] = df['stars'].apply(
            lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
        )
        
        return df
    
    @staticmethod
    def create_sample_dataset(n_samples: int = 100, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a sample dataset for testing.
        
        Args:
            n_samples (int): Number of samples to generate
            output_path (str, optional): Path to save dataset
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        sample_reviews = [
            "I love this product! It's absolutely amazing and exceeded my expectations.",
            "This is the worst purchase I've ever made. Terrible quality and broke immediately.",
            "It's okay, nothing special but does the job. Average experience.",
            "Fantastic! Best investment I've made. Highly recommend to everyone.",
            "Very disappointed. The description was misleading and customer service was rude.",
            "Great value for money. Fast shipping and good packaging.",
            "Not worth the price. Cheap materials and poor design.",
            "Absolutely wonderful! Makes me so happy every time I use it.",
            "Meh, it's fine I guess. Neither good nor bad.",
            "Horrible experience! I'm angry and want a refund immediately."
        ]
        
        # Generate variations
        import random
        random.seed(42)
        
        texts = []
        for _ in range(n_samples):
            base = random.choice(sample_reviews)
            # Add some randomization
            texts.append(base)
        
        df = pd.DataFrame({
            'text': texts,
            'source': 'sample',
            'id': range(n_samples)
        })
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Sample dataset saved to: {output_path}")
        
        return df


if __name__ == "__main__":
    # Test data loader
    print("Testing DataLoader...")
    
    # Create sample data
    sample_df = ReviewDataset.create_sample_dataset(n_samples=10)
    print(f"\nCreated sample dataset with {len(sample_df)} records")
    print(sample_df.head())
