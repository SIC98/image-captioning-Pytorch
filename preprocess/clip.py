from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch
import json

from cocodatasets import CocoCaptionsWithIds


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

dataset = CocoCaptionsWithIds(
    root="coco2017/train2017",
    annFile="coco2017/annotations/captions_train2017.json"
)

caption_id = 1
annotations = []

for id, img, text in tqdm(dataset):

    inputs = processor(
        text=text,
        images=img,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs)

    # this is the image-text similarity score
    logits_per_image = outputs.logits_per_image

    # we can take the softmax to get the label probabilities
    probs = logits_per_image.softmax(dim=1)[0]

    values, indices = torch.topk(probs, 4)

    new_text = [text[i] for i in indices]

    for i in new_text:
        annotaion = {
            "image_id": id,
            "id": caption_id,
            "caption": i
        }
        annotations.append(annotaion)
        caption_id += 1

caption = json.load(open("coco2017/annotations/captions_train2017.json", "r"))

caption["annotations"] = annotations

json.dump(caption, open(
    "coco2017/annotations/captions_train2017_clip_top4.json", "w")
)
