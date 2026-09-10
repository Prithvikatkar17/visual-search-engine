def search(query, model, tokenizer, image_vectors, valid_paths, top_k=6):
    from src.embed import embed_text
    text_features = embed_text(query, model, tokenizer)
    similarities = (image_vectors @ text_features.T).squeeze()
    top_indices = similarities.argsort(descending=True)[:top_k]
    return [(valid_paths[idx], similarities[idx].item()) for idx in top_indices]