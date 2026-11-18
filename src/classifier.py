## classifier and the triaing of the model

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from company_a import load_company_a, load_taxonomy
from sklearn.metrics.pairwise import cosine_similarity



def train_classifier(df, taxonomy):

    # Use included, labelled ERP records only
    train_df = df[
        (df["source"] == "ERP") &
        (df["include"] == 1) &
        (df["category_l1"].notna()) &
        (df["category_l2"].notna()) &
        (df["category_l3"].notna())
].copy()
    # Build text features first and
    train_df["text"] = (
        train_df["supplier"].fillna("") + " " +
        train_df["description"].fillna("") + " " +
        train_df["spend_type"].fillna("") + " " +
        train_df["material_type"].fillna("")
    )

    # Learn the complete category path
    train_df["category_path"] = (
        train_df["category_l1"] + "|||" +
        train_df["category_l2"] + "|||" +
        train_df["category_l3"]
    )

    X = train_df["text"]
    y = train_df["category_path"]

    #### Split the data  into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=None
    )

    #### Building  the model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Category paths: {y.nunique()}")
    print(f"Validation accuracy: {accuracy:.4f}")

    return model




## prdictor fucntion
def predict_categories(df, model):

    mask = (
    (df["include"] == 1) &
    (
        df["category_l1"].isna() |
        (df["category_l1"].astype(str).str.strip() == "")
    )
)

    if not mask.any():
        print("No eligible records require classification.")
        return df

    text = (
        df.loc[mask, "supplier"].fillna("") + " " +
        df.loc[mask, "description"].fillna("") + " " +
        df.loc[mask, "spend_type"].fillna("") + " " +
        df.loc[mask, "material_type"].fillna("")
    )

    predictions = model.predict(text)
    probabilities = model.predict_proba(text)

    confidence = probabilities.max(axis=1)


    predicted_categories = pd.Series(
        predictions,
        index=df.index[mask]
    ).str.split("|||", expand=True,regex=False)

    df.loc[mask, ["category_l1", "category_l2", "category_l3"]] = (
        predicted_categories.values
    )

    df.loc[mask, "confidence"] = confidence


    return df








## for company c generic template

def predict_zero_shot(df, taxonomy):

    ###2 Identify taxonomy levels dynamically
    category_columns = [
        col for col in taxonomy.columns
        if col.startswith("category_l")
    ]

    category_columns = sorted(
        category_columns,
        key=lambda col: int(col.split("_l")[1])
    )

    if not category_columns:
        raise ValueError("No category columns found in taxonomy")

    ###3 Prepare valid taxonomy paths
    taxonomy = taxonomy.dropna(
        subset=category_columns,
        how="all"
    ).copy()

    taxonomy["category_path"] = taxonomy[
        category_columns
    ].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

    taxonomy = taxonomy[
        taxonomy["category_path"] != ""
    ].reset_index(drop=True)

    if taxonomy.empty:
        raise ValueError("Taxonomy contains no valid category paths")

    ##### Prepare invoice text from available fields
    text_columns = [
        col for col in [
            "supplier",
            "description",
            "spend_type",
            "material_type"
        ]
        if col in df.columns
    ]

    if text_columns:
        df["classification_text"] = (
            df[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
    else:
        df["classification_text"] = ""

    #Identify eligible records
    eligible_mask = (
        df["include"].eq(1)
        if "include" in df.columns
        else pd.Series(True, index=df.index)
    )

    if not eligible_mask.any():
        return df.drop(columns=["classification_text"])

    invoice_text = df.loc[
        eligible_mask, "classification_text"
    ]

    #### Convert taxonomy and invoice text into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    combined_text = pd.concat(
        [
            taxonomy["category_path"],
            invoice_text
        ],
        ignore_index=True
    )

    all_vectors = vectorizer.fit_transform(combined_text)

    taxonomy_count = len(taxonomy)

    taxonomy_vectors = all_vectors[:taxonomy_count]
    invoice_vectors = all_vectors[taxonomy_count:]

    # Calculate similarity and select best categories
    similarities = cosine_similarity(
        invoice_vectors,
        taxonomy_vectors
    )

    best_matches = similarities.argmax(axis=1)
    confidence_scores = similarities.max(axis=1)

    # Assign matched taxonomy levels
    matched_categories = taxonomy.iloc[
        best_matches
    ][category_columns].reset_index(drop=True)

    for col in category_columns:
        if col not in df.columns:
            df[col] = None

        df[col] = df[col].astype(object)

        df.loc[eligible_mask, col] = (
            matched_categories[col].values
        )

    # Store similarity scores
    if "confidence" not in df.columns:
        df["confidence"] = None

    df.loc[eligible_mask, "confidence"] = confidence_scores

    # Remove temporary text column
    df.drop(columns=["classification_text"], inplace=True)

    return df
