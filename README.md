# Sentiment and Emotion Analysis on Consumer Reviews using NRCLex

This repository contains the implementation of sentiment and emotion analysis using the **NRCLex** lexicon-based approach, as presented in the paper:

**Sentiment and Emotion Analysis on Consumer Review using NRCLex**  
Muhammad Awais, Sara Durrani  
2nd International Conference on Engineering, Natural and Social Sciences (ICENSOS 2023), Konya, Turkey.

---

## Abstract

With the rapid growth of social media and online platforms, large volumes of unstructured consumer reviews are generated daily. Understanding sentiment polarity and emotional tone in such data is essential for market research, customer feedback analysis, and brand monitoring. This work applies the NRCLex lexicon-based method to classify consumer reviews into sentiment categories (positive, negative, neutral) and emotion classes such as joy, anger, fear, and sadness. The repository provides a clean, reproducible Python pipeline for text preprocessing, sentiment scoring, emotion extraction, and visualization.

---

## Methodology

1. Text preprocessing (noise removal, tokenization)
2. Sentiment analysis using NRCLex polarity scores
3. Emotion analysis using NRC emotion categories
4. Visualization of emotion distributions and sentiment trends

---

## Install dependencies

```bash
pip install -r requirements.txt
```
## Run sentiment and emotion analysis
```bash
python nrc_sentiment_analyzer.py
```
## Run visualization script
```bash
python visualization.py
```
## If your pipeline depends on data loading
```bash
python data_loader.py
```


