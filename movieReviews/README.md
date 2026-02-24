# 📊 Mini Project: Data Analysis from a CSV Dataset

A beginner‑friendly Python project designed to verify fundamental data‑analysis skills using **Pandas**, **NumPy**, and **Matplotlib**.  
This project walks through loading, cleaning, exploring, analyzing, and visualizing data from a CSV dataset.

---

## 📘 Purpose of the Project

The goal of this mini‑project is to practice essential data‑analysis concepts using real‑world style data.  
By completing it, you will demonstrate your ability to:

- Load and inspect CSV data  
- Clean and preprocess messy datasets  
- Perform descriptive statistics  
- Apply filtering, grouping, and sorting  
- Create data visualizations  
- Extract meaningful insights  
- Export processed data and results to files  
- Structure a small but complete analytical project  

---

## 📂 Dataset Description

You may use **any CSV dataset**, but this project assumes a **Movie Reviews Dataset** including columns like:

- `review_id`  
- `movie_title`  
- `review_text`  
- `rating` (e.g., 1–5 or 1–10)  
- `review_date`  
- `reviewer`  

If your dataset differs, feel free to adapt the steps accordingly.

---

## 🧪 Project Steps

### **1. Load & Inspect Data**
- Load using `pandas.read_csv()`
- Display:
  - First 10 rows  
  - Shape of the dataset  
  - Column names  
  - Data types via `df.info()`  
- Identify:
  - Missing values  
  - Duplicate rows  

---

### **2. Data Cleaning**
Operations include:

- Removing duplicate rows  
- Filling missing text fields with empty strings  
- Converting dates using `pd.to_datetime()`  
- Trimming whitespace in text fields  
- Ensuring numeric fields (e.g., rating) have correct data types  

---

### **3. Basic Statistical Analysis**

#### **Movie Review Statistics**
- Mean, median, min, max of ratings  
- Number of reviews per movie  
- Rating distribution  

#### **Optional: News Dataset Metrics**
- Articles per category  
- Articles per author  
- Most common publish day  
- Descriptive statistics for numeric columns  

---

### **4. Filtering & Grouping**

Examples implemented:

#### **For Movie Reviews**
- Reviews with rating ≥ 8  
- Top 5 most‑reviewed movies  
- Average rating per movie  

#### **For News Data (if applicable)**
- Technology articles  
- Most active author  
- Monthly article counts  

---

### **5. Data Visualization (Matplotlib)**

This project generates the following plots:

#### **Movie Reviews**
- Histogram → rating distribution  
- Bar chart → average rating per movie  
- Line plot → number of reviews over time  

#### **News Dataset (optional)**
- Articles per category  
- Articles per month  
- Histogram of article word counts  

All plots are saved as PNG files inside the `plots/` folder.

---

### **6. Optional Advanced Features**
For learners wanting to explore more:

- Word count column for each review  
- Extracting simple keywords  
- Detecting long or short reviews  
- Basic sentiment proxy based on rating threshold  

---

### **7. Exporting Results**
The script outputs:

- `cleaned_data.csv` → cleaned dataset  
- `summary.csv` → summary statistics or grouped results  
- PNG files of all generated plots  

---

## 📁 Expected Folder Structure

```
project/
│── data/
│    └── original.csv
│    └── cleaned_data.csv
│    └── summary.csv
│
│── plots/
│    └── rating_histogram.png
│    └── reviews_over_time.png
│
│── analysis.py
│── README.md
```

---

## 🧠 Sample Insights (Example)

Your results may vary, but typical insights include:

- **Highest‑rated movie:** *Inception*  
- **Most‑reviewed movie:** *The Dark Knight*  
- **Average rating across dataset:** 7.8  
- **Rating distribution:** Most users cluster around 7–9  
- **Review activity trend:** Steady increase from 2019–2024  

---

## ▶️ How to Run the Script

### **1. Install dependencies**

```bash
pip install pandas numpy matplotlib
```

### **2. Run the analysis**

```bash
python analysis.py
```

### **3. View outputs**
- Cleaned CSV: `data/cleaned_data.csv`
- Summary statistics: `data/summary.csv`
- Plots: `plots/*.png`  

---

## 🚀 What This Project Demonstrates

By completing this project, you show proficiency in:

- Loading, cleaning, and transforming CSV data  
- Working with **Pandas** DataFrames  
- Doing statistical analysis  
- Creating visualizations with **Matplotlib**  
- Structuring a mini data‑analysis pipeline  
- Exporting cleaned data and insights  

---
