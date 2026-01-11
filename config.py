class CFG: 
    # ASSIGNED TASK
    start = 1
    end = 7475

    # I/O DIRECTORIES
    folder_path_in = r"E:\VBS-DATA\keyframes\V3C"
    folder_path_out = r"/root/res/V3C"

    # BUILDLOADER PARAMETERS
    batch_size = 16 
    num_workers = 0

    # PADDLEOCR PARAMETERS 
    ocr_version = "PP-OCRv5"
    text_detection_model_name="PP-OCRv5_mobile_det"
    text_recognition_model_name="PP-OCRv5_mobile_rec"
    use_doc_orientation_classify=False
    use_doc_unwarping=False
    use_textline_orientation=False
    device="cpu"

    # VISUALIZATION OPTIONS
    visualized = False