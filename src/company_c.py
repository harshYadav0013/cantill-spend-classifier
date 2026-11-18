import pandas as pd
from classifier import predict_zero_shot

test_df = pd.DataFrame({
    "supplier": ["ABC Supplies", "XYZ Services"],
    "description": [
        "Industrial electrical switch",
        "Temporary labour staffing"
    ],
    "include": [1, 1]
})

test_taxonomy = pd.DataFrame({
    "category_l1": ["Materials", "Services"],
    "category_l2": ["Electrical", "Labour"],
    "category_l3": ["Switches", "Temporary Labour"],
    "category_l4": ["Industrial Switches", None]
})

result = predict_zero_shot(test_df, test_taxonomy)

print(result[[
    "description",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "confidence"
]].to_string(index=False))