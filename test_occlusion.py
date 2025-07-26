from ankigenerator.core.image_occlusion import batch_generate_image_occlusion_flashcards

creds_path = '/home/vidhidutta/.config/gcloud/ankigenerator-key.json'  # replace with your actual JSON path
cards = batch_generate_image_occlusion_flashcards(
    image_paths=['original_slide11_img3_1.png'],
    export_dir='out_cards',
    conf_threshold=70,
    max_masks=2,
    use_google_vision=True,
    credentials_path=creds_path
)
print(cards)
