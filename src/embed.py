import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    return model, preprocess, tokenizer

def embed_images_batched(image_paths, model, preprocess, batch_size=32):
    all_vectors = []
    valid_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors, batch_valid = [], []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_valid.append(path)
            except Exception:
                continue

        if not batch_tensors:
            continue

        batch_input = torch.stack(batch_tensors)
        with torch.no_grad():
            batch_features = model.encode_image(batch_input)
        batch_features = F.normalize(batch_features, dim=-1)

        all_vectors.append(batch_features)
        valid_paths.extend(batch_valid)

    return torch.cat(all_vectors, dim=0), valid_paths

def embed_text(query, model, tokenizer):
    text_input = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return F.normalize(text_features, dim=-1)