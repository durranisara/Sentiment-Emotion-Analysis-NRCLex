"""
Visualization Module
====================

Creates professional visualizations for sentiment and emotion analysis results.
Supports bar charts, pie charts, heatmaps, word clouds, and time series.

Author: Muhammad Awais
Email: mawaiskhan1808@gmail.com
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EmotionVisualizer:
    """
    Professional visualization tools for emotion and sentiment analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            figsize (tuple): Default figure size (width, height)
            dpi (int): Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'anger': '#e74c3c',
            'anticipation': '#f39c12',
            'disgust': '#8e44ad',
            'fear': '#2c3e50',
            'joy': '#f1c40f',
            'negative': '#c0392b',
            'positive': '#27ae60',
            'sadness': '#3498db',
            'surprise': '#e67e22',
            'trust': '#16a085',
            'neutral': '#95a5a6'
        }
    
    def plot_emotion_bar(
        self,
        emotions: Dict[str, float],
        title: str = "Emotion Analysis",
        top_n: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create bar chart of emotion scores.
        
        Args:
            emotions (dict): Emotion names and scores
            title (str): Chart title
            top_n (int, optional): Show only top N emotions
            save_path (str, optional): Path to save figure
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Sort and filter emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            sorted_emotions = sorted_emotions[:top_n]
        
        names, scores = zip(*sorted_emotions) if sorted_emotions else ([], [])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get colors
        bar_colors = [self.colors.get(name, '#3498db') for name in names]
        
        # Create bars
        bars = ax.bar(names, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )
        
        # Styling
        ax.set_xlabel('Emotions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(scores) * 1.15 if scores else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels if needed
        if len(names) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved emotion bar chart to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_sentiment_pie(
        self,
        sentiment: Dict[str, float],
        title: str = "Sentiment Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create pie chart for sentiment distribution.
        
        Args:
            sentiment (dict): Sentiment scores (positive, negative, neutral)
            title (str): Chart title
            save_path (str, optional): Path to save figure
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Filter out zero values
        labels = []
        sizes = []
        colors = []
        
        for label in ['positive', 'negative', 'neutral']:
            score = sentiment.get(label, 0)
            if score > 0:
                labels.append(label.capitalize())
                sizes.append(score)
                colors.append(self.colors.get(label, '#3498db'))
        
        if not sizes:
            labels = ['Neutral']
            sizes = [1]
            colors = [self.colors['neutral']]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(labels),
            shadow=True,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad
