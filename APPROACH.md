What I Did

I looked at how each company structured its data - The data came from different places and this had a problem - if I put their rows together, the spend totals were wrong.

Cleaning data and calculating spend

I made the columns have the same structure so the rest of the pipeline had the same field names to work with.

Company A had ERP amounts that were already without VAT but the amounts in the invoice tool had VAT included. I took VAT out of the records from the invoice tool. When a record had no VAT rate, I used 15 %.

For Company B I changed amounts into USD - I had currency rates to do this and I had a monthly FX file to use when currency rates weren't there.

Disallowed and Duplicates

I used explicit exclusion reasons, rather than just dropping records. This makes it simpler to understand why a transaction was not counted.

For Company A, I worked on duplicates in the invoice tool and matches in the ERP and invoice tool. Where both systems had the same purchase, I kept the ERP record.

Company B repeated records were identified using invoice and line details.

Matching supplier and item

I used the supplier and item ids when I could. I also used the supplied labels for Company A to map records to existing supplier and item keys.

I created deterministic keys from normalized text and hashing for records without a reliable match.


Classification

I predicted category paths for Company A and B using TF-IDF features with Logistic Regression on labeled records. I kept the entire category path while predicting, so the resulting levels are consistent.

We don’t have any labeled examples for Company C. I also added another zero-shot approach that compares invoice text to the available taxonomy paths using TF-IDF and cosine similarity.

Validation and constraints

I tested the supervised classifier using train/test split. Company A achieved about 96.8% validation accuracy on the category paths in that split.

The model can still make mistakes, especially for short or ambiguous descriptions . The zero-shot approach also depends on how closely invoice descriptions match the wording of the taxonomy.
