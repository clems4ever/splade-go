from sentence_transformers import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
# Run inference
queries = ["what causes aging fast"]
documents = [
    "UV-A light, specifically, is what mainly causes tanning, skin aging, and cataracts, UV-B causes sunburn, skin aging and skin cancer, and UV-C is the strongest, and therefore most effective at killing microorganisms. Again â\x80\x93 single words and multiple bullets.",
    "Answers from Ronald Petersen, M.D. Yes, Alzheimer's disease usually worsens slowly. But its speed of progression varies, depending on a person's genetic makeup, environmental factors, age at diagnosis and other medical conditions. Still, anyone diagnosed with Alzheimer's whose symptoms seem to be progressing quickly â\x80\x94 or who experiences a sudden decline â\x80\x94 should see his or her doctor.",
    "Bell's palsy and Extreme tiredness and Extreme fatigue (2 causes) Bell's palsy and Extreme tiredness and Hepatitis (2 causes) Bell's palsy and Extreme tiredness and Liver pain (2 causes) Bell's palsy and Extreme tiredness and Lymph node swelling in children (2 causes)",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 30522] [3, 30522]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)