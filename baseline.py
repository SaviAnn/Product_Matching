import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

def load_data():
    attributes_path = './data/train/attributes.parquet'
    resnet_path = './data/train/resnet.parquet'
    text_and_bert_path = './data/train/text_and_bert.parquet'
    train_path = './data/train/train.parquet'


    attributes = pd.read_parquet(attributes_path)
    resnet = pd.read_parquet(resnet_path)
    text_and_bert = pd.read_parquet(text_and_bert_path)
    train = pd.read_parquet(train_path)

    return attributes, resnet, text_and_bert, train

def extract_text_from_row(row):

    category_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['categories'].values())]
    )
    attributes_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['characteristic_attributes_mapping'].values())]
    )
    return f"{category_text} {attributes_text}"

def process_cat_and_attr(df):
    df['categories'] = df['categories'].apply(json.loads)
    df['characteristic_attributes_mapping'] = df['characteristic_attributes_mapping'].apply(json.loads)
    df['combined_text'] = df.apply(extract_text_from_row, axis=1)
    return df
# Удаление дублей
def delete_duples(df):
    df['sorted_variants'] = df[['variantid1', 'variantid2']].apply(lambda x: tuple(sorted(x)), axis=1)

    # Группируем по столбцу sorted_variants и нахождение групп дубликатов
    duplicate_groups = df[df.duplicated('sorted_variants', keep=False)]

    # Группируем строки по дубликатам и сохраняем индексы и значения target
    grouped_duplicates = duplicate_groups.groupby('sorted_variants').apply(lambda x: {
        'indices': list(x.index),
        'targets': x['target'].tolist()
    })

    # Фильтруем группы, где значения target отличаются
    indices_to_drop = []
    for group, info in grouped_duplicates.items():
        targets = info['targets']
        if len(set(targets)) > 1:  # Если в группе есть более одного уникального значения target
            indices_to_drop.extend(info['indices'])  

    # Удаляем строки с различными target и дубликаты из df
    df.drop(indices_to_drop, inplace=True)
    df.drop_duplicates(subset='sorted_variants', keep='first', inplace=True)

    # Удаляем вспомогательный столбец
    df.drop(columns=['sorted_variants'], inplace=True)

    # Обновленный df
    df.reset_index(drop=True, inplace=True)
    return df


def merge_data(train, resnet,attributes, text_and_bert):
    train_data = train.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(attributes[['variantid', 'combined_text']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'combined_text': 'text_2'})
    train_data = train_data.drop(columns=['variantid'])
    train_data = train_data.merge(text_and_bert[['variantid', 'name_bert_64', 'description']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'name_bert_64': 'name_1', 'description': 'description_1'})
    train_data = train_data.drop(columns=['variantid'])
    train_data = train_data.merge(text_and_bert[['variantid', 'name_bert_64', 'description']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'name_bert_64': 'name_2', 'description': 'description_2'})
    train_data = train_data.drop(columns=['variantid'])
    train_data = delete_duples(train_data)
    train_data.info()
    #train_data=train_data.dropna()
    return train_data

def combine_embeddings(row):
    pic_embeddings = np.concatenate([row['pic_embeddings_1'][0], row['pic_embeddings_2'][0]])
    text_embeddings = np.concatenate([row['text_embedding_1'], row['text_embedding_2']])
    return np.concatenate([pic_embeddings, text_embeddings])



def prepare_data(train_data, tfidf_vectorizer_text,tfidf_vectorizer_description): 
    train_data['text_1'] = train_data['text_1'].fillna('')
    train_data['text_2'] = train_data['text_2'].fillna('')
    train_data['description_1'] = train_data['description_1'].fillna('')
    train_data['description_2'] = train_data['description_2'].fillna('')
    train_data = train_data.reset_index(drop=True)
    text_data = train_data['text_1'] + ' ' + train_data['text_2']
    description_data = train_data['description_1'] + ' ' + train_data['description_2']
    text_embeddings = tfidf_vectorizer_text.fit_transform(text_data).toarray()
    description_embeddings = tfidf_vectorizer_description.fit_transform(description_data).toarray()

    train_data['combined_embeddings'] = train_data.apply(lambda row: np.concatenate([
        row['pic_embeddings_1'][0], row['pic_embeddings_2'][0],text_embeddings[row.name], description_embeddings[row.name],
        row['name_1'],row['name_2']
    ]), axis=1)

    X = np.vstack(train_data['combined_embeddings'].values)

    y = train_data['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24)

    return X_train, X_val, y_train, y_val, tfidf_vectorizer_text, tfidf_vectorizer_description




def train_model(X_train, y_train):
    bst = XGBClassifier(
    n_estimators=100,        
    max_depth=15,            
    learning_rate=0.1,      
    objective='binary:logistic',
    lambda_=0.1,            
    subsample=0.8,           
    )
    
    bst.fit(X_train, y_train)
    joblib.dump(bst, 'baseline_xgboost.pkl')
    
    return bst


def evaluate_model(model, X_val, y_val):
    y_preds = model.predict(X_val)
    precision, recall, _ = precision_recall_curve(y_val, y_preds)
    prauc = auc(recall, precision)
    print(f'PRAUC: {prauc}')

def main():
    attributes, resnet, text_and_bert, train = load_data()
    attributes = process_cat_and_attr(attributes)
    print('step 1')
    train_data = merge_data(train, resnet, attributes, text_and_bert)
    print('step 2')
    tfidf_vectorizer_text = TfidfVectorizer(max_features=1500)
    tfidf_vectorizer_description = TfidfVectorizer(max_features=1500)   
    print('step 3')
    X_train, X_val, y_train, y_val, tfidf_vectorizer_text, tfidf_vectorizer_description = prepare_data(train_data, tfidf_vectorizer_text,tfidf_vectorizer_description )
    print('step 4')

    model = train_model(X_train, y_train, X_val, y_val)
    print('step 5')
    evaluate_model(model, X_val, y_val)
    print('step 6')
    joblib.dump(tfidf_vectorizer_text, 'vectorizer_text1500.pkl')
    joblib.dump(tfidf_vectorizer_description , 'vectorizer_description1500.pkl')
    print('step 7')

if __name__ == "__main__":
    main()
