# Data Decisions

Some choices I made while building the pipeline:

- Company A VAT: ERP values are net. Invoice-tool values are gross. If the rate is missing, 15% VAT is assumed.
- A purchases: When the same purchase occurs in both systems the record in the ERP is kept and the invoice-tool record is set to 'DUPLICATE'.
- Invoice-tool copies: duplicate invoice records are flagged based on invoice number, supplier, description and amount.
- Company B currency: USD amounts are unchanged. Other currencies are translated at the rate provided or using the monthly FX rates.
- The monthly FX file is used where a non-USD transaction does not have a rate in the invoice data.
- Exclusions: Excluded records are included in the output with an exclusion reason and zero spend.
- Supplier and item keys: Pre-existing identifiers and supplied labels are better. Hash keys as a fallback.
- Categorisation: Supervised classification is done in the presence of labelled data. In case of Company C, as labelled data is not available, taxonomy similarity categorisation is used.
- The score stored for cosine similarity is a matching score, not a probability.

I felt comfortable using conservative matching and explicit reasons for exclusion, rather than throwing away data or assuming that would add to the spend total.