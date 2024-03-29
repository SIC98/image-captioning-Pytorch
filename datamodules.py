from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from cocodatasets import CaptionDataset


class COCODataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 128
        self.valid_batch_size = 100
        self.num_workers = 4

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.train_dataset = CaptionDataset(
            root="coco2017/train2017",
            annFile="coco2017/annotations/captions_train2017_clip_top4.json",
            transform=transform,
            cpi=4
        )

        self.val_dataset = CaptionDataset(
            root="coco2017/val2017",
            annFile="coco2017/annotations/captions_val2017.json",
            transform=transform,
            cpi=5
        )

    def train_dataloader(self):

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.valid_batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False
        )


if __name__ == "__main__":
    datamodule = COCODataModule()
    dataloader = datamodule.train_dataloader()
    print(dataloader)
    for img, text in dataloader:
        # torch.Size([16, 3, 256, 256])
        print(img.shape)
        # 5 16 two cats sitting on a lounge chair arm looking out a window
        print(len(text), len(text[0]), text[0][0])
        break
