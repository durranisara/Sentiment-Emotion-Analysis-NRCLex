

import re
import string
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter
import warnings

# Suppress NLTK download messages
warnings.filterwarnings('ignore')

try:
    from nrclex import NRCLex
except ImportError:
    raise ImportError(
        "NRCLex is required. Install it using: pip install nrclex"
    )


class NRCSentimentAnalyzer:
    """
    A professional sentiment and emotion analyzer using NRCLex.
    
    This class provides comprehensive text analysis capabilities including:
    - Sentiment classification (positive, negative, neutral)
    - Emotion detection (anger, fear, joy, sadness, surprise, trust, etc.)
    - Raw emotion scoring
    - Batch processing capabilities
    - Visualization data preparation
    
    Attributes:
        text (str): The input text to analyze
        nrc_object (NRCLex): The NRCLex object for analysis
    """
    
    # Emotion categories supported by NRCLex
    EMOTION_CATEGORIES = [
        'anger', 'anticipation', 'disgust', 'fear', 
        'joy', 'negative', 'positive', 'sadness', 
        'surprise', 'trust'
    ]
    
    # Sentiment categories
    SENTIMENT_CATEGORIES = ['positive', 'negative', 'neutral']
    
    def __init__(self, text: str):
        """
        Initialize the analyzer with text data.
        
        Args:
            text (str): The text to analyze
            
        Raises:
            ValueError: If text is empty or None
            TypeError: If text is not a string
        """
        if text is None:
            raise ValueError("Text cannot be None")
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
            
        self.original_text = text
        self.text = self._preprocess(text)
        self.nrc_object = NRCLex(self.text)
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text by removing noise and normalizing.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove mentions and hashtags (optional)
        4. Remove extra whitespace
        5. Remove punctuation (optional, configurable)
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment(self) -> Dict[str, Union[str, float]]:
        """
        Get overall sentiment analysis.
        
        Returns:
            Dict containing:
                - 'label': Overall sentiment label (positive/negative/neutral)
                - 'score': Sentiment score (-1.0 to 1.0)
                - 'confidence': Confidence level (0.0 to 1.0)
        """
        scores = self.nrc_object.affect_frequencies
        
        positive_score = scores.get('positive', 0)
        negative_score = scores.get('negative', 0)
        
        # Calculate net sentiment
        net_sentiment = positive_score - negative_score
        
        # Determine label
        if net_sentiment > 0.05:
            label = 'positive'
        elif net_sentiment < -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence
        total_sentiment = positive_score + negative_score
        confidence = min(total_sentiment * 2, 1.0) if total_sentiment > 0 else 0.0
        
        return {
            'label': label,
            'score': round(net_sentiment, 4),
            'confidence': round(confidence, 4),
            'positive_score': round(positive_score, 4),
            'negative_score': round(negative_score, 4)
        }
    
    def get_emotions(self) -> Dict[str, float]:
        """
        Get all emotion scores from the text.
        
        Returns:
            Dict[str, float]: Dictionary of emotion names and their scores (0.0 to 1.0)
        """
        return {
            emotion: round(self.nrc_object.affect_frequencies.get(emotion, 0.0), 4)
            for emotion in self.EMOTION_CATEGORIES
        }
    
    def get_raw_emotion_scores(self) -> Dict[str, int]:
        """
        Get raw emotion scores (word counts) from the text.
        
        This method returns the actual count of words associated with each emotion
        category in the text.
        
        Returns:
            Dict[str, int]: Dictionary of emotion names and their raw word counts
        """
        raw_scores = {}
        words = self.nrc_object.words
        
        for emotion in self.EMOTION_CATEGORIES:
            count = sum(1 for word in words if emotion in self.nrc_object.lexicon.get(word, []))
            raw_scores[emotion] = count
        
        return raw_scores
    
    def get_top_emotions(self, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top n emotions from the text.
        
        Args:
            n (int): Number of top emotions to return (default: 3)
            
        Returns:
            List[Tuple[str, float]]: List of (emotion, score) tuples sorted by score
        """
        emotions = self.get_emotions()
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:n]
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        Get the single most dominant emotion in the text.
        
        Returns:
            Tuple[str, float]: (emotion_name, score)
        """
        top = self.get_top_emotions(1)
        return top[0] if top else ('neutral', 0.0)
    
    def get_affect_words(self) -> Dict[str, List[str]]:
        """
        Get words associated with each emotion category.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping emotions to lists of words
        """
        affect_dict = {}
        words = self.nrc_object.words
        
        for emotion in self.EMOTION_CATEGORIES:
            emotion_words = [
                word for word in words 
                if emotion in self.nrc_object.lexicon.get(word, [])
            ]
            if emotion_words:
                affect_dict[emotion] = emotion_words
        
        return affect_dict
    
    def get_sentences_with_emotions(self) -> List[Dict]:
        """
        Analyze emotions at the sentence level.
        
        Returns:
            List[Dict]: List of dictionaries containing sentence and its emotions
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(self.original_text)
        sentence_analysis = []
        
        for sent in sentences:
            if sent.strip():
                analyzer = NRCSentimentAnalyzer(sent)
                sentence_analysis.append({
                    'sentence': sent,
                    'sentiment': analyzer.get_sentiment(),
                    'emotions': analyzer.get_emotions(),
                    'dominant_emotion': analyzer.get_dominant_emotion()
                })
        
        return sentence_analysis
    
    def get_word_emotion_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of each word to its associated emotions.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping words to their emotions
        """
        mapping = {}
        for word in self.nrc_object.words:
            emotions = self.nrc_object.lexicon.get(word, [])
            if emotions:
                mapping[word] = emotions
        return mapping
    
    def analyze(self) -> Dict:
        """
        Perform complete analysis of the text.
        
        Returns:
            Dict: Comprehensive analysis results
        """
        return {
            'text': self.original_text[:200] + '...' if len(self.original_text) > 200 else self.original_text,
            'sentiment': self.get_sentiment(),
            'emotions': self.get_emotions(),
            'raw_scores': self.get_raw_emotion_scores(),
            'top_emotions': self.get_top_emotions(),
            'dominant_emotion': self.get_dominant_emotion(),
            'affect_words': self.get_affect_words(),
            'word_count': len(self.nrc_object.words),
            'unique_words': len(set(self.nrc_object.words))
        }
    
    @staticmethod
    def batch_analyze(texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of analysis results
        """
        results = []
        for i, text in enumerate(texts):
            try:
                analyzer = NRCSentimentAnalyzer(text)
                result = analyzer.analyze()
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'text': text[:50] if text else 'None'
                })
        return results


def quick_analyze(text: str) -> Dict:
    """
    Quick analysis function for simple use cases.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict: Simplified analysis results
    """
    analyzer = NRCSentimentAnalyzer(text)
    sentiment = analyzer.get_sentiment()
    dominant = analyzer.get_dominant_emotion()
    
    return {
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score'],
        'dominant_emotion': dominant[0],
        'emotion_score': dominant[1]
    }


if __name__ == "__main__":
    # Example usage
    sample_text = "I love this product! It's amazing and makes me so happy."
    
    print("=" * 60)
    print("NRCLex Sentiment and Emotion Analyzer")
    print("=" * 60)
    
    analyzer = NRCSentimentAnalyzer(sample_text)
    results = analyzer.analyze()
    
    print(f"\nText: {results['text']}")
    print(f"\nSentiment: {results['sentiment']['label']} "
          f"(score: {results['sentiment']['score']})")
    print(f"Dominant Emotion: {results['dominant_emotion'][0]} "
          f"(score: {results['dominant_emotion'][1]})")
    print("\nAll Emotions:")
    for emotion, score in results['emotions'].items():
        if score > 0:
            print(f"  - {emotion}: {score}")
