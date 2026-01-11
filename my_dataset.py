import cv2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(list(Path(image_dir).glob("*.jpg")))
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            return None
        return str(path), img