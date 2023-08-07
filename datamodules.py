from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

from cocodatasets import CaptionDataset
from utils import collate_fn


class COCODataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 80
        self.valid_batch_size = 80
        self.num_workers = 4

        with open('wordmap.json', 'r') as j:
            self.word_map = json.load(j)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.train_dataset = CaptionDataset(
            root='coco2014/train2014',
            annFile='coco2014/caption_datasets/captions_train2014.json',
            transform=transform,
            cpi=5
        )

        self.val_dataset = CaptionDataset(
            root='coco2014/val2014',
            annFile='coco2014/caption_datasets/captions_val2014.json',
            transform=transform,
            cpi=5
        )

        self.test_dataset = CaptionDataset(
            root='coco2014/val2014',
            annFile='coco2014/caption_datasets/captions_test2014.json',
            transform=transform,
            cpi=5
        )

    def setup(self, stage=None):
        pass

    def train_dataloader(self):

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            # collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.valid_batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=collate_fn
        )


if __name__ == '__main__':
    datamodule = COCODataModule()
    dataloader = datamodule.train_dataloader()
    print(dataloader)
    for img, text in dataloader:
        # torch.Size([16, 3, 256, 256])
        print(img.shape)
        # 5 16 two cats sitting on a lounge chair arm looking out a window
        print(len(text), len(text[0]), text[0][0])
        break
