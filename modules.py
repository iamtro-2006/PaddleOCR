from my_dataset import ImageDataset
from torch.utils.data import DataLoader 

def collate_fn(batch):
    return [item for item in batch if item is not None]

def build_loader(image_dir, batch_size, num_workers, shuffle=False):
    inference_dataset = ImageDataset(image_dir)
    inference_loader = DataLoader(inference_dataset, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  shuffle=shuffle)
    return inference_loader, len(inference_dataset) 