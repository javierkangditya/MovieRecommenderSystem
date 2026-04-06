# =========================================================
# RECOMMENDER SYSTEM PIPELINE
# Data Cleaning → EDA → Matrix → Modeling → Evaluation
# =========================================================

# =========================================================
# 0. LIBRARIES
# =========================================================
library(tidyverse)
library(data.table)
library(Matrix)
library(recommenderlab)
library(ggplot2)

# =========================================================
# 1. DATA LOADING & CLEANING
# =========================================================

ratings_raw <- fread("Data/Raw/rating.csv")
movies_raw  <- fread("Data/Raw/movie.csv")

cat("===== RAW DATA OVERVIEW =====\n")
print(dim(ratings_raw))
print(dim(movies_raw))

ratings <- unique(na.omit(ratings_raw))
movies  <- unique(na.omit(movies_raw))

ratings[, userId := as.integer(userId)]
ratings[, movieId := as.integer(movieId)]
ratings[, rating  := as.numeric(rating)]

movies[, movieId := as.integer(movieId)]
movies[, title   := as.character(title)]
movies[, genres  := as.character(genres)]

# FILTER SPARSITY
user_counts  <- ratings[, .N, by = userId]
movie_counts <- ratings[, .N, by = movieId]

ratings <- ratings[
  userId %in% user_counts[N >= 10, userId] &
    movieId %in% movie_counts[N >= 5, movieId]
]

cat("\n===== CLEANED DATA =====\n")
print(dim(ratings))
print(dim(movies))

saveRDS(ratings, "Data/Processed/ratings_clean.rds")
saveRDS(movies,  "Data/Processed/movies_clean.rds")

# =========================================================
# 2. EDA
# =========================================================

ratings <- readRDS("Data/Processed/ratings_clean.rds")
movies  <- readRDS("Data/Processed/movies_clean.rds")

ratings <- ratings %>%
  filter(!is.na(userId), !is.na(movieId), !is.na(rating))

cat("===== DATA SUMMARY =====\n")
cat("Users:", n_distinct(ratings$userId), "\n")
cat("Movies:", n_distinct(ratings$movieId), "\n")
cat("Ratings:", nrow(ratings), "\n")

# Rating Distribution
p1 <- ggplot(ratings, aes(rating)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue") +
  theme_minimal()

ggsave("Outputs/Figures/rating_dist.png", p1)

# User activity
user_activity <- ratings %>% count(userId)

p2 <- ggplot(user_activity, aes(n)) +
  geom_histogram(binwidth = 10, fill = "darkgreen") +
  theme_minimal()

ggsave("Outputs/Figures/user_activity.png", p2)

# Sparsity
n_users  <- n_distinct(ratings$userId)
n_movies <- n_distinct(ratings$movieId)
n_ratings <- nrow(ratings)

sparsity <- 1 - (n_ratings / (n_users * n_movies))

cat("\nSparsity:", sparsity, "\n")

write.csv(
  data.frame(
    Users = n_users,
    Movies = n_movies,
    Ratings = n_ratings,
    Sparsity = sparsity
  ),
  "Outputs/Tables/eda_summary.csv",
  row.names = FALSE
)

# =========================================================
# 3. USER-ITEM MATRIX & SPLIT
# =========================================================

ratings <- readRDS("Data/Processed/ratings_clean.rds")

user_index  <- as.numeric(factor(ratings$userId))
movie_index <- as.numeric(factor(ratings$movieId))

rating_matrix <- sparseMatrix(
  i = user_index,
  j = movie_index,
  x = ratings$rating
)

rating_rrm <- as(rating_matrix, "realRatingMatrix")

set.seed(123)

eval_scheme <- evaluationScheme(
  rating_rrm,
  method = "split",
  train = 0.8,
  given = 10,
  goodRating = 4
)

saveRDS(eval_scheme, "Data/Processed/eval_scheme.rds")

# =========================================================
# 4. MODELING
# =========================================================

eval_scheme <- readRDS("Data/Processed/eval_scheme.rds")

train  <- getData(eval_scheme, "train")
known  <- getData(eval_scheme, "known")
unknown <- getData(eval_scheme, "unknown")

n <- min(100, nrow(known))

known_small   <- known[1:n, ]
unknown_small <- unknown[1:n, ]

# MODELS
model_pop <- Recommender(train, method = "POPULAR")

model_mf <- tryCatch({
  Recommender(train, method = "LIBMF",
              parameter = list(dim = 10, iter = 30, lambda = 0.01))
}, error = function(e) {
  Recommender(train, method = "SVD", parameter = list(k = 10))
})

train_small <- train[1:min(500, nrow(train)), ]

model_knn <- Recommender(train_small, method = "IBCF", parameter = list(k = 30))

# PREDICTIONS
pred_pop <- predict(model_pop, known_small, n = 10)
pred_mf  <- predict(model_mf,  known_small, n = 10)
pred_knn <- predict(model_knn, known_small, n = 10)

# =========================================================
# 5. METRICS
# =========================================================

recall_at_k <- function(pred, actual, k = 10) {
  pred_list <- as(pred, "list")
  scores <- c()
  
  for (i in seq_along(pred_list)) {
    if (i > length(actual)) break
    p <- pred_list[[i]][1:k]
    a <- actual[[i]]
    if (length(a) == 0) next
    scores <- c(scores, length(intersect(p, a)) / length(a))
  }
  
  if (length(scores) == 0) return(0)
  mean(scores)
}

map_at_k <- function(pred, actual, k = 10) {
  pred_list <- as(pred, "list")
  ap <- c()
  
  for (i in seq_along(pred_list)) {
    if (i > length(actual)) break
    
    p <- pred_list[[i]][1:k]
    a <- actual[[i]]
    
    if (length(a) == 0) next
    
    hit <- 0
    sum_prec <- 0
    
    for (j in seq_along(p)) {
      if (p[j] %in% a) {
        hit <- hit + 1
        sum_prec <- sum_prec + hit / j
      }
    }
    
    if (hit > 0) {
      ap <- c(ap, sum_prec / min(length(a), k))
    }
  }
  
  if (length(ap) == 0) return(0)
  mean(ap)
}

dcg <- function(rel) {
  sum(rel / log2(seq_along(rel) + 1))
}

ndcg_at_k <- function(pred, actual, k = 10) {
  pred_list <- as(pred, "list")
  scores <- c()
  
  for (i in seq_along(pred_list)) {
    if (i > length(actual)) break
    
    p <- pred_list[[i]][1:k]
    a <- actual[[i]]
    
    if (length(a) == 0) next
    
    rel <- as.numeric(p %in% a)
    ideal <- sort(rel, decreasing = TRUE)
    
    if (dcg(ideal) == 0) next
    
    scores <- c(scores, dcg(rel) / dcg(ideal))
  }
  
  if (length(scores) == 0) return(0)
  mean(scores)
}

actual <- lapply(1:nrow(unknown_small), function(i) {
  user <- as(unknown_small[i, ], "list")[[1]]
  names(user)
})

results <- data.frame(
  model = c("POPULAR", "MF", "KNN (IBCF)"),
  
  recall_at_10 = c(
    recall_at_k(pred_pop, actual),
    recall_at_k(pred_mf, actual),
    recall_at_k(pred_knn, actual)
  ),
  
  map_at_10 = c(
    map_at_k(pred_pop, actual),
    map_at_k(pred_mf, actual),
    map_at_k(pred_knn, actual)
  ),
  
  ndcg_at_10 = c(
    ndcg_at_k(pred_pop, actual),
    ndcg_at_k(pred_mf, actual),
    ndcg_at_k(pred_knn, actual)
  )
)

print(results)

write.csv(results, "Outputs/Tables/model_results.csv", row.names = FALSE)

# =========================================================
# 6. VISUALIZATION
# =========================================================

results_long <- results %>%
  pivot_longer(cols = -model, names_to = "metric", values_to = "value")

p <- ggplot(results_long, aes(model, value, fill = model)) +
  geom_bar(stat = "identity") +
  facet_wrap(~metric, scales = "free") +
  theme_minimal()

ggsave("Outputs/Figures/model_comparison.png", p)
print(p)
