# Product Matching
Product matching is crucial for various marketplaces, enabling buyers to choose similar items with different delivery options or prices. It also helps suggest alternative products that closely match the one being considered by the user. During a hackathon, participants were provided with parquet files (which cannot be shared here), and the approach described is conceptual in nature.

Initially, a logistic regression architecture combined with TF-IDF vectorizers for text data was chosen based on the provided dataset. Ideally, BERT embeddings would have been utilized, but due to inference limitations imposed by the hackathon rules, TF-IDF vectorizers remained in the final solution.

The task of determining whether two products are identical was formulated as a binary classification problem. The overall architecture is illustrated in the diagrams below.

Several data processing experiments were conducted, including calculating cosine similarity for specific features before concatenation to reduce the model's complexity and feature count. This optimization provided a performance boost, along with the implementation of XGBoost using hyperparameters tuned through Grid Search.

In conclusion, solving this problem was both challenging and rewarding, showcasing the potential for innovation in product matching systems.
![Alt text](Baseline_part1.png)
![Alt text](Baseline_part2.png)
