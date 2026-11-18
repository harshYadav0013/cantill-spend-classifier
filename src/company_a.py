import pandas as pd
import hashlib


## non procurement keywords
NON_PROCUREMENT_KEYWORDS = [
    "INTERCOMPANY",
    "REVENUE",
    "ROYALT",
    "PARTNER BILLING",
    "FINANCING COST"
]



## loading func for the erp file
def load_erp():

    df = pd.read_csv("data/company_A/erp_goods_receipts.csv",
        low_memory=False)

    return df


    
    
## loading fun for the invoice file
def load_invoice_tool():

    df = pd.read_csv("data/company_A/invoice_tool_export.csv",low_memory=False)

    return df




## loading func for the labels file
def load_labels():

    df = pd.read_csv("data/company_A/labels_A.csv",low_memory=False)

    return df

## loading func for the taxonomy files
def load_taxonomy():

    df = pd.read_csv("data/company_A/taxonomy_A.csv",low_memory=False)

    return df





## normalize the vat

def normalize_vat(vat):

    if isinstance(vat, str):
        vat = vat.replace("%", "").strip()

    vat = float(vat)

    if vat > 1:
        vat = vat / 100

    return vat






## fun for the stadardization
def standardize_erp(df):

    clean_df = df.rename(columns={
        "line_id": "record_id",
        "supplier_name": "supplier",
        "material_short_text": "description",
        "value_zar": "amount",
        "po_number": "po_reference",
        "posting_date": "transaction_date"
    }).copy()

    clean_df["source"] = "ERP"
    clean_df["vat_rate"] = 0
 

    clean_df["record_id"] = "ERP:" + clean_df["record_id"].astype(str)

    clean_df = clean_df[[
    "record_id",
    "supplier",
    "description",
    "amount",
    "quantity",
    "source",
    "vat_rate",
    "po_reference",
    "po_item",
    "transaction_date",
    "vendor_id",
    "spend_type",
    "service_short_text",
    "material_number",
    "material_type"
]]


    return clean_df




def standardize_invoice(df):


    clean_df = df.rename(columns={
        "tool_record_id":"record_id",
        "supplier_name_as_captured": "supplier",
        "line_text": "description",
        "gross_amount": "amount",
        "quantity":"quantity",
        "po_reference":"po_reference",
        "invoice_date":"transaction_date",
        "invoice_number":"invoice_number",
        "status":"status"
    }).copy()

    clean_df["source"] = "TOOL"
    clean_df["vat_rate"] = (
    df["vat_rate"]
    .fillna("15%")
    .apply(normalize_vat)
)

    clean_df["record_id"] = "TOOL:" + clean_df["record_id"].astype(str)

    clean_df = clean_df[[
        "record_id",
        "supplier",
        "description",
        "amount",
        "quantity",
        "source",
        "vat_rate",
        "po_reference",
        "transaction_date",
        "invoice_number",
        "status"
    ]]

    return clean_df


## non precurement data handling 
def apply_non_procurement(df):
    pattern = "|".join(NON_PROCUREMENT_KEYWORDS)

    mask = (
        df["description"].str.contains(pattern, na=False) |
        df["supplier"].str.upper().str.contains(pattern, na=False)
    )

    df.loc[
        df["exclusion_reason"].isna() & mask,
        "exclusion_reason"
    ] = "NON_PROCUREMENT"

    return df


## normalize the po refrences

def normalize_reference(value):
    if pd.isna(value):
        return None

    value = str(value).upper().strip()
    value = value.replace("PO", "").replace("#", "").strip()

    return value

## normalized the invoice
def normalize_invoice(value):
    if pd.isna(value):
        return None

    value = str(value).upper().strip()
    value = " ".join(value.split())

    return value



## ckeck duplicates within the invoice files
def apply_tool_duplicates(df):
    mask = (
        (df["source"] == "TOOL") &
        df.duplicated(
            subset=[
                "invoice_number",
                "supplier",
                "description",
                "amount"
            ],
            keep="first"
        )
    )

    df.loc[
        mask & df["exclusion_reason"].isna(),
        "exclusion_reason"
    ] = "DUPLICATE"

    return df



## func to check the duplicate in both erp and the invoice files compared useng po refrece,supplier and description
def apply_cross_system_duplicates(df):

    erp = df[df["source"] == "ERP"].copy()

    tool = df[
        (df["source"] == "TOOL") &
        (df["exclusion_reason"].isna())
    ].copy()

    #### Normalizing the  identifiers
    for data in [erp, tool]:

        data["po_key"] = (
            data["po_reference"]
            .fillna("")
            .astype(str)
            .str.split("/")
            .str[0]
        )

        data["supplier_key"] = (
            data["supplier"]
            .fillna("")
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )

        data["description_key"] = (
            data["description"]
            .fillna("")
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )

    # Remove records with missing matching identifiers
    erp = erp[
        (erp["po_key"] != "") &
        (erp["supplier_key"] != "") &
        (erp["description_key"] != "")
    ]

    tool = tool[
        (tool["po_key"] != "") &
        (tool["supplier_key"] != "") &
        (tool["description_key"] != "")
    ]

    # Match invoice records against ERP records
    matches = tool.reset_index().merge(
        erp,
        on=["po_key", "supplier_key", "description_key"],
        suffixes=("_tool", "_erp")
    )

    # Allow small rounding differences
    duplicate_ids = matches.loc[
        (matches["amount_tool"] - matches["amount_erp"]).abs() <= 1,
        "index"
    ].unique()

    # Mark TOOL records as duplicates
    df.loc[
        duplicate_ids,
        "exclusion_reason"
    ] = "DUPLICATE"

    return df







## func to assing the keys
def assign_keys(df):

    #### Normalizing supplier names
    df["supplier_norm"] = (
        df["supplier"]
        .fillna("")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    # Normalize descriptions
    df["description_norm"] = (
        df["description"]
        .fillna("")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    #### Use ERP records as the source of known keys
    erp = df[df["source"] == "ERP"]

    supplier_lookup = (
        erp.groupby("supplier_norm")["supplier_key"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .to_dict()
    )

    item_lookup = (
        erp.groupby(["supplier_norm", "description_norm"])["item_key"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .to_dict()
    )

    # Assign keys to records that do not aleady have them
    missing_supplier = df["supplier_key"].isna()
    missing_item = df["item_key"].isna()

    df.loc[missing_supplier, "supplier_key"] = (
        df.loc[missing_supplier, "supplier_norm"]
        .map(supplier_lookup)
    )

    df.loc[missing_item, "item_key"] = [
        item_lookup.get((supplier, description))
        for supplier, description in zip(
            df.loc[missing_item, "supplier_norm"],
            df.loc[missing_item, "description_norm"]
        )
    ]

    return df









### for unknown keys crete a deterministic key
def assign_fallback_keys(df):

    for index, row in df.iterrows():

        if pd.isna(row["supplier_key"]):
            supplier_value = row["supplier_norm"] or row["record_id"]

            supplier_hash = hashlib.sha1(
                supplier_value.encode()
            ).hexdigest()[:7]

            df.at[index, "supplier_key"] = "SG-" + supplier_hash

        if pd.isna(row["item_key"]):
            item_value = (
                row["supplier_norm"] + "|" + row["description_norm"]
                if row["description_norm"]
                else row["supplier_norm"] + "|" + row["record_id"]
            )

            item_hash = hashlib.sha1(
                item_value.encode()
            ).hexdigest()[:7]

            df.at[index, "item_key"] = "IG-" + item_hash

    return df








## combining the data

def load_company_a():

    df1 = load_erp()
    df2 = load_invoice_tool()
    labels = load_labels()

    df1 = standardize_erp(df1)
    df2 = standardize_invoice(df2)

    combined = pd.concat([df1,df2],ignore_index=True)
    combined = combined.merge(
    labels,
    on="record_id",
    how="left"
)
    combined["description"] = (
    combined["description"]
    .fillna("")
    .str.strip()
    .str.upper()
)


    ## exclusion reason to find if purchase happen or not
    combined["exclusion_reason"]=None
    combined.loc[
    (combined["source"] == "TOOL") & (combined["status"] == "Cancelled"),
    "exclusion_reason"
] = "CANCELLED"

    combined.loc[
    (combined["source"] == "TOOL") &
    (
        (combined["status"] == "On Hold") |
        (combined["status"] == "Rejected")
    ),
    "exclusion_reason"
] = "NOT_APPROVED"

    ## cal the vat if source is the tool only

    combined["amount"] = pd.to_numeric(
    combined["amount"].astype(str).str.replace(",", ""),
    errors="coerce"
)

    combined["original_amount"] = combined["amount"]


    
    combined["amount"] = combined.apply(
    lambda row: row["amount"] / (1 + row["vat_rate"])
    if row["source"] == "TOOL"
    else row["amount"],
    axis=1
)
    combined = apply_non_procurement(combined)
    combined = apply_tool_duplicates(combined)
    combined = apply_cross_system_duplicates(combined)

    #### Determine which records are eligible for spend
    combined["include"] = (
    combined["exclusion_reason"].isna().astype(int)
)

# Excluded records must contribute zero spend
    combined["spend"] = combined["amount"].where(
    combined["include"] == 1,
    0
).fillna(0)

#### clean transaction dates
    combined["transaction_date"] = pd.to_datetime(
    combined["transaction_date"],
    errors="coerce"
)


    ##### asinging the keys in the last
    combined = assign_keys(combined)
    combined = assign_fallback_keys(combined)

    return combined





