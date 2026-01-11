import os
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR
from my_dataset import ImageDataset
from config import CFG
from modules import build_loader, collate_fn
 

def main():
    folder = Path(CFG.folder_path_in)

    model = PaddleOCR(ocr_version=CFG.ocr_version,
                      text_detection_model_name=CFG.text_detection_model_name,
                      text_recognition_model_name=CFG.text_recognition_model_name,
                      use_doc_orientation_classify=CFG.use_doc_orientation_classify,
                      use_doc_unwarping=CFG.use_doc_unwarping,
                      use_textline_orientation=CFG.use_textline_orientation,
                      device=CFG.device,
                    )
    
    for image_dir in tqdm(sorted(folder.iterdir()), desc="Folder Processing"):
        if not image_dir.is_dir():
            continue
        
        image_dir_name = image_dir.name
        
        if int(image_dir_name) < CFG.start or int(image_dir_name) > CFG.end :
            continue
        
        dataloader, total_images = build_loader(image_dir=image_dir,
                                                batch_size=CFG.batch_size,
                                                num_workers=CFG.num_workers)
        if total_images == 0:
            continue

        results_to_save = []
        start = time.time()
        
        for batch in tqdm(dataloader, desc=f"OCR {image_dir_name}", leave=False):
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
                if CFG.visualized:
                    res.save_to_img(f"{CFG.folder_path_out}/visuals/{image_dir_name}/{filename}.jpg")

        output_file = f"{CFG.folder_path_out}/{image_dir_name}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        elapsed = round(time.time() - start, 2)
        print(f"\nFinished {image_dir_name}: {elapsed}s for {total_images} images.")

if __name__ == "__main__":
    main()