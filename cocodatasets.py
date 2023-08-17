from torch.utils.data import Dataset
from torchvision import datasets


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, root, annFile, transform, cpi):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.dataset = datasets.CocoCaptions(
            root=root,
            annFile=annFile,
            transform=transform
        )

        self.dataset_size = len(self.dataset) * cpi
        self.cpi = cpi

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.dataset[i // self.cpi][0]

        caption = self.dataset[i // self.cpi][1][i % self.cpi]

        all_captions = self.dataset[i // self.cpi][1][:self.cpi]

        return img, caption, all_captions

    def __len__(self):
        return self.dataset_size


class CocoCaptionsWithIds(datasets.CocoCaptions):

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return id, image, target
