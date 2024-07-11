from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import tarfile
import io
from tqdm import tqdm

class RefL4Dataset(Dataset):
    def __init__(self, dataset_path, split, images_file='images.tar.gz', custom_transforms=None):
        """
        Initialize the RefL4Dataset class.

        Parameters:
        - dataset_path (str): Path to the dataset directory.
        - split (str): Dataset split, typically "val", "test", or "all".
        - images_file (str): Name of the tar file containing images, default to 'images.tar.gz'.
        - custom_transforms: Custom image transformations to apply, default to 'None'.
        """
        super(RefL4Dataset, self).__init__()
        assert split in ['val', 'test', 'all'], 'split should be val, test or all'
        self.dataset_path=dataset_path
        self.split = split
        self.images_file=images_file
        self.transforms = custom_transforms
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = load_dataset(self.dataset_path)
        all_splits=concatenate_datasets([self.dataset['val'],self.dataset['test']])
        self.dataset['all']=all_splits
        self.images=self._load_images_from_tar()

    def _load_images_from_tar(self):
        images = {}
        print(f"Loading images from {self.dataset_path}/{self.images_file}")
        with tarfile.open(f"{self.dataset_path}/{self.images_file}", "r:gz") as tar:
            for member in tqdm(tar.getmembers()):
                if member.isfile() and member.name.endswith(('jpg', 'jpeg', 'png', 'webp')):
                    f = tar.extractfile(member)
                    if f:
                        image = Image.open(io.BytesIO(f.read()))
                        # transfer the grayscale image to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        # remove any previous extension of name
                        
                        images[member.name] = image
        return images
    
    def change_split(self, split):
        assert split in ['val', 'test', 'all'], 'split should be val, test or all'
        self.split = split

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        """
        Returns:
        - image (Tensor): Transformed image data.
        - data (dict): Other sample data.
        """
        data = self.dataset[self.split][idx]
        image = self.images[data['file_name']]
        if self.transforms:
            image = self.transforms(image)
        return image, data

# Example usage:
if __name__ == '__main__':
    custom_transforms = None
    ref_l4_dataset = RefL4Dataset('/Users/jchen12/Documents/misc/Ref-L4', split='all', custom_transforms=custom_transforms)
    print(len(ref_l4_dataset))
    print(ref_l4_dataset[0])
    # swith to val split
    ref_l4_dataset.change_split('val')
    print(len(ref_l4_dataset))
    print(ref_l4_dataset[0])
    # swith to test split
    ref_l4_dataset.change_split('test')
    print(len(ref_l4_dataset))
    print(ref_l4_dataset[0])

    ref_l4_dataset.change_split('all')
    print(len(ref_l4_dataset))
    print(ref_l4_dataset[0])
