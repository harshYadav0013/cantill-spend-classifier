import pandas as pd
import hashlib
from classifier import train_classifier, predict_categories

## load func for the comapnay b
def load_b_data():

    df = pd.read_csv(
        "data/company_B/invoice_lines.csv",
        low_memory=False
    )

    labels = pd.read_csv(
        "data/company_B/seed_labels.csv"
    )

    taxonomy = pd.read_csv(
        "data/company_B/taxonomy_B.csv"
    )

    fx = pd.read_csv(
        "data/company_B/fx_rates_usd.csv"
    )

    return df, labels, taxonomy, fx




## func to standardizeing the data
def standardize_company_b(df):

    clean_df = df.rename(columns={
        "line_id": "record_id",
        "vendor_id": "supplier",
        "line_description": "description",
        "line_amount": "amount",
        "po_number": "po_reference",
        "po_line": "po_item",
        "posting_date": "transaction_date",
        "gl_account_description": "spend_type",
        "supplier_type": "material_type"
    }).copy()

    # Common fields used by our classifier
    clean_df["source"] = "ERP"
    clean_df["vat_rate"] = 0

    clean_df["amount"] = pd.to_numeric(
        clean_df["amount"], errors="coerce"
    )

    clean_df["description"] = (
        clean_df["description"].fillna("").str.strip().str.upper()
    )

    return clean_df



## currency conversion
def convert_to_usd(df, fx):

    df = df.copy()

    df["month"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    ).dt.strftime("%Y-%m")

    df["currency_rate"] = pd.to_numeric(
        df["currency_rate"], errors="coerce"
    )

    fx = fx.copy()
    fx["month"] = fx["month"].astype(str)

    df = df.merge(
        fx,
        on=["currency", "month"],
        how="left"
    )

    ##### USD transactions are already  convereted
    df.loc[df["currency"] == "USD", "currency_rate"] = 1

    # Use monthly FX when the transaction rate is missing
    df["currency_rate"] = df["currency_rate"].fillna(
        df["units_per_usd"]
    )

    # Convert local currency to US dolar
    df["amount"] = (
        df["amount"] / df["currency_rate"]
    )

    df.drop(
        columns=["units_per_usd", "month"],
        inplace=True
    )

    return df


## exclusion and  non procurement 

def apply_b_eligibility(df, labels):

    df = df.merge(
        labels[[
            "line_id",
            "include",
            "exclusion_reason",
            "category_l1",
            "category_l2",
            "category_l3"
        ]],
        left_on="record_id",
        right_on="line_id",
        how="left"
    )

    df.drop(columns=["line_id"], inplace=True)

    # Unlabelled records are provisionally included
    df["include"] = df["include"].fillna(1)

    ##### Identifying the  non-procurement GL accounts
    non_procurement_accounts = [
        "GL40035",
        "GL40039"
    ]

    mask = (
        df["gl_account"].isin(non_procurement_accounts) &
        df["exclusion_reason"].isna()
    )

    df.loc[mask, "include"] = 0
    df.loc[mask, "exclusion_reason"] = "NON_PROCUREMENT"

    
    df["include"] = df["include"].astype(int)

    return df



### generate the keys andassing them fucns
def generate_hash(value, prefix):

    hashed = hashlib.sha1(
        str(value).encode()
    ).hexdigest()[:7]

    return prefix + "-" + hashed


def assign_b_keys(df):

    df = df.copy()

    # Normalize supplier identifier
    df["supplier_norm"] = (
        df["supplier"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Generate supplier keys
    df["supplier_key"] = [
        generate_hash(
            supplier if supplier else record_id,
            "SG"
        )
        for supplier, record_id in zip(
            df["supplier_norm"],
            df["record_id"]
        )
    ]

    # Prefer material number for item identity
    material = (
        df["material_number"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    description = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    item_identity = [
        material_value if material_value
        else description_value if description_value
        else record_id
        for material_value, description_value, record_id
        in zip(material, description, df["record_id"])
    ]

    df["item_key"] = [
        generate_hash(
            supplier + "|" + item,
            "IG"
        )
        for supplier, item in zip(
            df["supplier_norm"],
            item_identity
        )
    ]

    return df












## duplicate remover func

def apply_b_duplicates(df):

    duplicate_columns = [
        "invoice_number",
        "invoice_line",
        "supplier",
        "currency",
        "amount",
        "description"
    ]

    mask = (
        (df["include"] == 1) &
        df.duplicated(
            subset=duplicate_columns,
            keep="first"
        )
    )

    df.loc[mask, "include"] = 0
    df.loc[mask, "exclusion_reason"] = "DUPLICATE"

    print("Duplicate records excluded:", mask.sum())

    return df



if __name__ == "__main__":

    # 1. Load Company B data
    df, labels, taxonomy, fx = load_b_data()

    # 2. Standardize invoice records
    combined = standardize_company_b(df)

    # 3. Convert amounts to USD
    combined = convert_to_usd(combined, fx)

    # 4. Apply eligibility rules and attach seed labels
    combined = apply_b_eligibility(combined, labels)

    # 5. Remove duplicate invoice lines
    combined = apply_b_duplicates(combined)

    # 6. Generate supplier and item keys
    combined = assign_b_keys(combined)

    # 7. Train classifier
    model = train_classifier(combined, taxonomy)

    # 8. Predict missing categories
    combined = predict_categories(combined, model)

    # 9. Inspect predictions
    print("\nCATEGORY COUNTS")
    print(combined["category_l1"].value_counts())

    print("\nUNCLASSIFIED INCLUDED RECORDS")
    print(
        combined.loc[
            (combined["include"] == 1) &
            (combined["category_l1"].isna())
        ].shape[0]
    )

    print("\nSAMPLE PREDICTIONS")
    print(combined[[
        "record_id",
        "description",
        "category_l1",
        "category_l2",
        "category_l3",
        "confidence"
    ]].head(15).to_string(index=False))


# Final spend: excluded records must have zero spend
combined["spend"] = combined["amount"].where(
    combined["include"] == 1, 0
).fillna(0)

# Clear categories for excluded records
excluded = combined["include"] == 0
combined.loc[
    excluded, ["category_l1", "category_l2", "category_l3"]
] = None

# Select final output columns
output = combined[[
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "category_l1",
    "category_l2",
    "category_l3",
    "confidence"
]]

output.to_csv("outputs/company_B__B.csv", index=False)

print("Company B output saved successfully!")