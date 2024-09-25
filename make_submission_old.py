import pandas as pd
import numpy as np
import joblib
from baseline import process_cat_and_attr
from sklearn.decomposition import PCA
from os import path
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.field_decoding import decode_categories, decode_attributes
nltk.download('punkt_tab')
nltk.download('stopwords')
def load_test_data():
    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    val_path = './data/test/test.parquet'

    attributes = pd.read_parquet(attributes_path, engine='pyarrow')
    resnet = pd.read_parquet(resnet_path, engine='pyarrow')
    text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')
    test = pd.read_parquet(val_path, engine='pyarrow')
    
    return attributes, resnet, text_and_bert, test

import joblib
from sklearn.preprocessing import LabelEncoder

encoders_path = "./utils/category_encoder_%s.pkl"

def get_encoders_paths():
    return map(lambda i: encoders_path % i, range(4))

# Функция для загрузки готовых энкодеров
def load_encoders():
    encoders = []
    for encoder_path in get_encoders_paths():
        encoders.append(joblib.load(encoder_path))
    return encoders
nonword_regex = re.compile(r"[^\w+]")
russian_stopwords = stopwords.words("russian")
def extract_text_from_row(row):

    # category_text = " ".join(row["categories"])
    attributes_text = row["characteristic_attributes_mapping"]
    # return f"{category_text} {attributes_text}"
    return attributes_text




def clear_description(description: str) -> str:
    return " ".join(
        filter(
            lambda w: w not in russian_stopwords,
            word_tokenize(nonword_regex.sub(" ", description.lower())),
        )
    )

# Функция для обработки категориальных данных и атрибутов
def process_cat_and_attr(df, encoders=[]):
    df["categories"] = df["categories"].apply(decode_categories)
    for i in range(4):
        df[f"category_{i}"] = df["categories"].apply(lambda c: c[i])

    # Используем уже загруженные энкодеры
    for i, e in enumerate(encoders):
        df[f"category_{i}"] = df[f"category_{i}"].apply(
            lambda c: next(iter(e.transform((c,)) or [0]))
        )

    df["characteristic_attributes_mapping"] = df["characteristic_attributes_mapping"].apply(decode_attributes)
    df.drop("categories", axis=1, inplace=True)
    df["combined_text"] = df.apply(extract_text_from_row, axis=1)
    df["combined_text"] = df["combined_text"].fillna(" ").apply(clear_description)
    return df
def prepare_text_and_bert(df: pd.DataFrame) -> pd.DataFrame:
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    df["descritption"] = df["description"].fillna("").apply(clear_description)
    return df


def merge_data(test, resnet, attributes, text_and_bert):
    test_data = test.merge(
        resnet[["variantid", "main_pic_embeddings_resnet_v1"]],
        left_on="variantid1",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(
        columns={"main_pic_embeddings_resnet_v1": "pic_embeddings_1"}
    )
    test_data = test_data.drop(columns=["variantid"])

    test_data = test_data.merge(
        resnet[["variantid", "main_pic_embeddings_resnet_v1"]],
        left_on="variantid2",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(
        columns={"main_pic_embeddings_resnet_v1": "pic_embeddings_2"}
    )
    test_data = test_data.drop(columns=["variantid"])

    test_data = test_data.merge(
        attributes[["variantid", "combined_text","category_0", "category_1","category_2", "category_3"]],
        left_on="variantid1",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(columns={"combined_text": "text_1", "category_0": "cat0_1",
                                             "category_1":"cat1_1","category_2": "cat2_1", "category_3": "cat_3_1"})
    test_data = test_data.drop(columns=["variantid"])


    test_data = test_data.merge(
        attributes[["variantid", "combined_text","category_0", "category_1","category_2", "category_3"]],
        left_on="variantid2",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(columns={"combined_text": "text_2","category_0": "cat0_2",
                                             "category_1":"cat1_2","category_2": "cat2_2", "category_3": "cat_3_2"})
    test_data = test_data.drop(columns=["variantid"])
    test_data = test_data.merge(
        text_and_bert[["variantid", "name_bert_64", "description"]],
        left_on="variantid1",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(
        columns={"name_bert_64": "name_1", "description": "description_1"}
    )
    test_data = test_data.drop(columns=["variantid"])
    test_data = test_data.merge(
        text_and_bert[["variantid", "name_bert_64", "description"]],
        left_on="variantid2",
        right_on="variantid",
        how="left",
    )
    test_data = test_data.rename(
        columns={"name_bert_64": "name_2", "description": "description_2"}
    )
    test_data.info()
    # test_data=test_data.dropna()
    return test_data

def prepare_test_data(test_data,  tfidf_vectorizer_text,tfidf_vectorizer_description):
    # Замена пропусков на пустые строки
    test_data["text_1"] = test_data["text_1"].fillna("")
    test_data["text_2"] = test_data["text_2"].fillna("")
    test_data["description_1"] = test_data["description_1"].fillna("")
    test_data["description_2"] = test_data["description_2"].fillna("")
    test_data = test_data.reset_index(drop=True)
    text_data = test_data["text_1"] + " " + test_data["text_2"]
    description_data = test_data["description_1"] + " " + test_data["description_2"]
    text_embeddings = tfidf_vectorizer_text.fit_transform(text_data).toarray()
    description_embeddings = tfidf_vectorizer_description.fit_transform(
        description_data
    ).toarray()
    # Уменьшаем размерность text_embeddings до 100
    pca_text = PCA(n_components=100)
    text_embeddings_reduced = pca_text.fit_transform(text_embeddings)

    # Уменьшаем размерность description_embeddings до 100
    pca_description = PCA(n_components=100)
    description_embeddings_reduced = pca_description.fit_transform(description_embeddings)
    print('PCA done!')
    test_data["combined_embeddings"] = test_data.apply(
        lambda row: np.concatenate(
            [
                row["pic_embeddings_1"][0],
                row["pic_embeddings_2"][0],
               text_embeddings_reduced[row.name],
                description_embeddings_reduced[row.name],
                row["name_1"],
                row["name_2"],
                np.array([row["cat0_1"]]), # Обернуть скаляры в массивы
                np.array([row["cat0_2"]]),
                np.array([row["cat1_1"]]),
                np.array([row["cat1_2"]]),
                np.array([row["cat2_1"]]),
                np.array([row["cat2_2"]]),
                np.array([row["cat_3_1"]]),
                np.array([row["cat_3_2"]]),


            ]
        ),
        axis=1,
    )
    X_test = np.vstack(test_data['combined_embeddings'].values)
    return X_test

def main():
    attributes, resnet, text_and_bert, test = load_test_data()
    print('check 1')
    encoders = load_encoders()
    print('check 2')
    attributes = process_cat_and_attr(attributes,  encoders=encoders)
    print('check 3')
    text_and_bert = prepare_text_and_bert(text_and_bert)
    print('check 4')
    test_data = merge_data(test, resnet, attributes, text_and_bert)
    print('check 5')
    tfidf_vectorizer_text = joblib.load('vectorizer_text1400_cat.pkl')
    tfidf_vectorizer_description = joblib.load('vectorizer_description1400_cat.pkl')

    X_test = prepare_test_data(test_data, tfidf_vectorizer_text, tfidf_vectorizer_description )
    print('check 6')
    model = joblib.load('baseline_xgboost_cat_full.pkl')
    
    predictions_prob = model.predict_proba(X_test)[:, 1]
    predictions = (predictions_prob >= 0.5).astype(int)

    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        'target': predictions
    })
    
    submission.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    main()
