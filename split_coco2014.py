import json

captions_val2014 = 'coco2014/annotations/captions_val2014.json'
captions_test2014 = 'coco2014/annotations/captions_test2014.json'
dataset_coco = 'coco2014/caption_datasets/dataset_coco.json'


with open(dataset_coco, 'r') as j:
    split_dataset = json.load(j)

with open(captions_val2014, 'r') as j:
    coco = json.load(j)

val_cocoid = [i['cocoid']
              for i in split_dataset['images'] if i['split'] == 'val']
val_images = [i for i in coco['images'] if i['id'] in val_cocoid]
val_captions = [i for i in coco['annotations'] if i['image_id'] in val_cocoid]

coco['images'] = val_images
coco['annotations'] = val_captions

with open(captions_val2014, 'w') as j:
    json.dump(coco, j)

test_cocoid = [i['cocoid']
               for i in split_dataset['images'] if i['split'] == 'test']
test_images = [i for i in coco['images'] if i['id'] in test_cocoid]
test_captions = [i for i in coco['annotations']
                 if i['image_id'] in test_cocoid]

coco['images'] = test_images
coco['annotations'] = test_captions

with open(captions_test2014, 'w') as j:
    json.dump(coco, j)
