dish_list = [
    'kadhi_pakoda', 'aloo_tikki', 'butter_chicken', 'aloo_methi', 'sandesh',
    'bhatura', 'bandar_laddu', 'dum_aloo', 'jalebi', 'lassi',
    'qubani_ka_meetha', 'chapati', 'kofta', 'double_ka_meetha', 'kalakand',
    'sutar_feni', 'gavvalu', 'cham_cham', 'mysore_pak', 'aloo_gobi',
    'misi_roti', 'kachori', 'shankarpali', 'biryani', 'bhindi_masala',
    'chicken_razala', 'poornalu', 'malapua', 'litti_chokha', 'maach_jhol',
    'kajjikaya', 'chikki', 'daal_puri', 'rasgulla', 'kadai_paneer',
    'paneer_butter_masala', 'modak', 'unni_appam', 'pithe', 'adhirasam',
    'sohan_halwa', 'boondi', 'basundi', 'dal_makhani', 'chicken_tikka',
    'shrikhand', 'aloo_matar', 'sheer_korma', 'lyangcha',
    'chicken_tikka_masala', 'chana_masala', 'naan', 'ghevar', 'ariselu',
    'dal_tadka', 'karela_bharta', 'palak_paneer', 'pootharekulu',
    'navrattan_korma', 'chak_hao_kheer', 'ras_malai', 'gulab_jamun',
    'kakinada_khaja', 'imarti', 'gajar_ka_halwa', 'dharwad_pedha', 'poha',
    'sheera', 'rabri', 'phirni', 'sohan_papdi', 'aloo_shimla_mirch',
    'makki_di_roti_sarson_da_saag', 'kuzhi_paniyaram', 'misti_doi', 'doodhpak',
    'daal_baati_churma', 'anarsa', 'chhena_kheeri', 'ledikeni'
]



N_IMAGE = 1000

from icrawler.builtin import BingImageCrawler
import os, time

def download_images(keyword, target_num=N_IMAGE, out_dir='dataset', filters=None):
    """Download at least target_num images for a keyword into out_dir/keyword"""
    save_dir = os.path.join(out_dir, keyword)
    os.makedirs(save_dir, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': save_dir},
                               parser_threads=2,
                               downloader_threads=4)

    def count_images():
        return len([f for f in os.listdir(save_dir)
                    if f.lower().endswith(('jpg','jpeg','png'))])

    current = count_images()
    if current >= target_num:
        print(f"[{keyword}] already has {current} images")
        return
    keyword = keyword.replace(' ','_')
    # Try with larger max_num to compensate for failures
    attempts = 0
    while current < target_num and attempts < 5:
        attempts += 1
        remaining = target_num - current
        print(f"[{keyword}] Attempt {attempts} – need {remaining} more images…")
        crawler.crawl(keyword=keyword.replace(' ','_'),
                      max_num=remaining*2,  # overshoot to offset bad URLs
                      filters=filters)

        # Wait a little to avoid throttling
        time.sleep(3)
        current = count_images()
        print(f"[{keyword}] now has {current} images")


    print(f"[{keyword}] Done. Final count = {current}")

for dish in dish_list:
    download_images(dish, target_num=N_IMAGE, out_dir='food_images')

