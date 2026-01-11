import os
import cv2
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader 
from paddleocr import PaddleOCR 

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

def main():
    folder = Path(r"E:\VBS-DATA\keyframes\V3C")
    batch_size = 16 
    num_workers = 0 


    model = PaddleOCR(ocr_version="PP-OCRv5",
                      text_detection_model_name="PP-OCRv5_mobile_det",
                      text_recognition_model_name="PP-OCRv5_mobile_rec",
                      use_doc_orientation_classify=False,
                      use_doc_unwarping=False,
                      use_textline_orientation=False,
                      device="cpu",
                    )
    
    for image_dir in tqdm(sorted(folder.iterdir()), desc="Folder Processing"):
        if not image_dir.is_dir():
            continue
        
        dataloader, total_images = build_loader(image_dir=image_dir,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
        if total_images == 0:
            continue

        results_to_save = []
        start = time.time()
        
        for batch in tqdm(dataloader, desc=f"OCR {image_dir.name}", leave=False):
            if not batch: 
                continue

            paths, imgs = zip(*batch)
            results = model.predict(list(imgs))
             
            if results is None: 
                continue

            for path, res in zip(paths, results):
                filename = Path(path).stem
                texts = []
                if res:
                    texts = res["rec_texts"]
                results_to_save.append([filename, texts])
                res.save_to_img(f"/root/res/visuals/{filename}.jpg")

        output_file = f"/root/res/V3C/{image_dir.name}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        elapsed = round(time.time() - start, 2)
        print(f"\nFinished {image_dir.name}: {elapsed}s for {total_images} images.")

if __name__ == "__main__":
    main()