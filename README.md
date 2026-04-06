# MovieRecommendationSystem

## Project Overview
This project builds and evaluates a movie recommendation system using collaborative filtering approaches. The system analyzes user-movie interaction data and compares multiple recommendation algorithms using ranking-based evaluation metrics such as Recall@10, MAP@10, and NDCG@10.

## Folder Structure
- `Data/Raw` : original datasets (ratings and movie metadata)
- `Data/Processed` : cleaned datasets and intermediate objects (RDS files)
- `Scripts` : R scripts for full pipeline (cleaning, EDA, modeling, evaluation)
- `Outputs/Figures` : visualizations from EDA and model comparison
- `Outputs/Tables` : summary statistics and evaluation results
- `Outputs/Models` : trained recommendation models saved as RDS

## Pipeline
1. Data Cleaning  
   - Remove missing values and duplicates  
   - Convert data types to appropriate formats  
   - Filter sparse users (≥10 ratings) and sparse movies (≥5 ratings)

2. Exploratory Data Analysis (EDA)  
   - Rating distribution analysis  
   - User activity distribution  
   - Top-rated and most popular movies  
   - Dataset sparsity calculation  

3. User-Item Matrix Construction  
   - Convert ratings into sparse matrix format  
   - Transform into recommenderlab realRatingMatrix  

4. Train-Test Split  
   - Evaluation scheme using 80/20 split  
   - Top-N recommendation setting (N = 10)  
   - Threshold-based relevance (rating ≥ 4)

5. Modeling  
   - Popularity-based recommender (POPULAR)  
   - Matrix Factorization (LIBMF / SVD fallback)  
   - Item-Based Collaborative Filtering (IBCF)

6. Evaluation  
   - Recall@10  
   - MAP@10 (Mean Average Precision)  
   - NDCG@10 (Normalized Discounted Cumulative Gain)

## How to Run
1. Open RStudio project  
2. Install required packages:
   tidyverse  
   data.table  
   Matrix  
   recommenderlab  
   ggplot2  

3. Run scripts in order:
   1_data_cleaning.R  
   2_eda.R  
   3_matrix_split.R  
   4_modeling.R  
   5_evaluation_visualization.R  

4. All outputs will be saved automatically in the Outputs folder

## Key Insights
- Dataset shows strong popularity bias where a small number of movies dominate interactions
- POPULAR model consistently outperforms MF and IBCF across all metrics
- Matrix Factorization performs moderately but is limited by data sparsity
- Item-based CF struggles in sparse environments
- Ranking metrics (MAP@10 and NDCG@10) are more informative than Recall alone

## Evaluation Summary
- POPULAR model achieved the best MAP@10 and NDCG@10
- MF shows moderate but unstable performance
- IBCF performs the weakest due to sparsity sensitivity

## Recommendations
- Use POPULAR baseline as benchmark for sparse datasets
- Improve MF via hyperparameter tuning (latent factors, regularization, iterations)
- Consider hybrid models (POPULAR + MF) for better performance
- Collect more interaction data to reduce sparsity issues

## Dataset
MovieLens dataset by GroupLens Research  
https://grouplens.org/datasets/movielens/
