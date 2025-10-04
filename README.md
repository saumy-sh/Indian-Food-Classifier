## Indian Food Predictor(Image based)
Helps in predicting the major Indian dishes out of the 72 major dishes it was trained on across India.

Checkout the site [here](https://what-is-that.streamlit.app/)

A simple streamlit based interface where you can either upload or take picture of your dish and then prediction is shown.
Here are some images of the simple interface.
<img width="1034" height="785" alt="ui" src="https://github.com/user-attachments/assets/d9c83ff0-9dd8-4bab-a124-07337764474d" />
<img width="1034" height="874" alt="prediction" src="https://github.com/user-attachments/assets/963ba6d2-c60f-4f23-931c-eb10a300ec39" />

The food classes are:
[
 'lassi', 'dal_tadka', 'malapua', 'poornalu', 'chana_masala',
 'sohan_papdi', 'shankarpali', 'shrikhand', 'aloo_gobi', 'phirni',
 'rabri', 'poha', 'bhatura', 'double_ka_meetha', 'dal_makhani',
 'lyangcha', 'kalakand', 'pootharekulu', 'sandesh', 'aloo_tikki',
 'unni_appam', 'kachori', 'chicken_tikka_masala', 'sutar_feni',
 'paneer_butter_masala', 'biryani', 'doodhpak', 'mysore_pak', 'chikki',
 'gavvalu', 'dharwad_pedha', 'jalebi', 'misi_roti', 'karela_bharta',
 'bandar_laddu', 'daal_baati_churma', 'dum_aloo', 'chak_hao_kheer',
 'sheer_korma', 'bhindi_masala', 'ghevar', 'misti_doi', 'aloo_matar',
 'ras_malai', 'butter_chicken', 'imarti', 'chicken_tikka',
 'makki_di_roti_sarson_da_saag', 'kadai_paneer', 'qubani_ka_meetha',
 'gajar_ka_halwa', 'maach_jhol', 'sohan_halwa', 'navrattan_korma',
 'gulab_jamun', 'litti_chokha', 'basundi', 'boondi', 'chicken_razala',
 'palak_paneer', 'naan', 'rasgulla', 'kofta', 'sheera', 'modak',
 'kakinada_khaja', 'ariselu', 'daal_puri', 'kadhi_pakoda', 'chapati',
 'aloo_shimla_mirch', 'aloo_methi'
]

## Training strategy
# Pre-training
I initially trained a ResNet50 model on Food101 dataset with 101 classes so that model can learn basic features of dishes. I used this dataset as it had around 69k training images and around 22k validation images which helped model generalise.

# Fine-tuning
Then I created a local dataset containig 80 dishes with around 150-200 images each class. After filtering, I was left with 72 classes with approximately same number of images per class.
I then fine-tuned my pre-trained model by changing classifier head with a 72 classifier head.Below are training and validation accuracy and loss for 30 epochs.
<img width="1500" height="500" alt="loss-accuracy" src="https://github.com/user-attachments/assets/9cc8e022-19c9-4658-86ba-e3a997137eb7" />

# Usecase
I plan to add a RAG or word-extraction based system to get recipe from predicted class.
