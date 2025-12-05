# COVID-19 Trajectory and Sentiment Analysis Project

This repository contains our work on analyzing the evolution of COVID-19 pandemic trajectories and public emotions on Twitter.  
The project combines **epidemiological clustering**, **time-series and cross-sectional features**, and **sentiment + topic modeling of social media text**.

---

## 🚀 Quick Start

To reproduce the main results used in our report:

**➡ Run `Group12_final.ipynb` directly**  
This notebook contains the full pipeline for epidemic trajectory clustering and visualization.

---

## 🧠 Full Project Execution Instructions

### 1. Epidemic Trajectory & Cross-Sectional Feature Analysis  
Run the notebook:  Group12_final.ipynb

---

### 2. Sentiment and Topic Modeling (Full Workflow)

Navigate to the `sentiment/` folder.  
Run scripts **in numerical stage order**:

| File / Stage | Description |
|---|---|
| `first_crawl.py` | Tweet crawling *(may be unstable due to network/API limits)* |
| `second_wordcloud.py` | Word cloud preprocessing |
| `third_tweets.py` | Text cleaning & formatting |
| `forth_embedding.py` | Generate tweet embeddings |
| `fifth.py` | Data merging & preparation |
| `sixth_emotion.py` | Sentiment classification (RoBERTa + GoEmotions) |
| `seven_newtopiccluster.py` | Topic modeling using BERTopic |
| `eight_stage.py` | Visualization and result interpretation |
| `stage1-3`, `stage3-5`, `stage6-9`, `stage10-2` | Intermediate processing results |

⚠ Notes:  
- Crawling may fail occasionally — please retry when necessary.  
- Sentiment and topic modeling take **long computation time**, especially embedding and clustering.  

---

## 👥 Contribution

| Member | Contribution |
|---|---|
| **Xiaosa Liu** | Overall design, all code in sentiment, integration of all code |
| **Lingxiao Yin** | Graph-based clustering |
| **Jia Wei** | Clustering analysis implementation |
| **Chuanqi Zhou** | Cross-sectional and time-series feature analysis |
| **Chenxi Cai** | General clustering |

---

## Additional Notes

- Some original tweet data may not be publicly distributable. Sample and intermediate data are provided when possible.  
- This repository is intended for research and reproducibility purposes.

---

### AI Declaration

AI tools (ChatGPT) were used for learning related knowledge, writing refinement, and code debugging support.  
All analyses and results were produced and validated by the authors.

---

Feel free to open issues or requests if you need guidance running the project.


