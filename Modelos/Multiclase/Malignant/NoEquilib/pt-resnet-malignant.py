# %% [markdown]
# ## Entrenamiento de ResNet50 con data augmentation y 3 modelos
# 
# Se procede al particionado de datos de train y valid por separado para aplicar data augmentation en entrenamiento mediante la librería imgaug. Luego, se proceden a realizar un modelo de clasificación binaria para discriminar primero el tipo de enfermedad, y posteriormente, se creará un modelo específico para enfermedades cancerígenas y no cancerígenas.

# %% [markdown]
# Comenzamos por la importación de librerías

# %%
# Librerías utilizadas por el script

import os
os.environ['HF_HOME'] = '/mnt/homeGPU/hexecode/cache/'

import timm
import numpy as np
import pandas as pd
import cv2
import torch
import fastbook
import fastai
import fastcore
import PIL
import shutil
import albumentations

import matplotlib.pyplot as plt
import imgaug as ia
import seaborn as sns

from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fastai.vision.all import *
#from nbdev.showdoc import *
from fastai.vision.all import *
import torch.utils.mobile_optimizer as mobile_optimizer
from torch.utils.mobile_optimizer import optimize_for_mobile

"""
!pip install -Uqq fastbook
!pip install nbdev
"""
fastbook.setup_book()
torch.cuda.is_available()

# %% [markdown]
# A continuación, se declaran las constantes usadas como parámetros a las funciones

# %%
# Número de clases. 2 para la binaria.
NUM_CLASSES = 2
TRAIN_DIR = "malignantThumbnails/"
SEED = 26
# Número de canales de cada imagen
CHANNELS = 3
IMAGE_RESIZE = 512
RESIZE_METHOD = 'squish'
LOSS_FUNC = FocalLossFlat()
METRICS = [accuracy, BalancedAccuracy(), error_rate, Recall(average = 'weighted'), Precision(average = 'weighted')]
VALID_PCT = 0.3

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 40
FREEZE_EPOCHS = 80

EARLY_STOP_PATIENCE = 3

BATCH_SIZE = 32

train_malignant = pd.read_csv('train_malignant.csv')


#valid = pd.read_csv('valid.csv')
#aug_df = pd.concat([train_aug,valid])


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)

# %% [markdown]
# Ahora, podemos volver a calcular la proporción:

# %%
#train_malignant = train.loc[train['bin'] == 1]
#train_malignant['label'] = train_malignant['label'].replace({'melanoma_metastasis': 'melanoma'})
#train_malignant.to_csv('train_malignant.csv',index=False)

# %%
print(train_malignant.label.value_counts())
sns.countplot(train_malignant, x='label', order=train_malignant.label.value_counts().index)

# %% [markdown]
# Ahora, movemos estas imágenes a una nueva carpeta para poder realizar el entrenamiento sin interferencia del resto. El objetivo es conseguir crear un modelo especializado únicamente en clases positivas en cáncer. En la celda anterior, además, se han agrupado las clases melanoma y melanoma infiltrante, ya que ambas clases pertenecen a la misma enfermedad, siendo el infiltrante una versión más virulenta y avanzada. Como se trataba únicamente de 23 ejemplares, se ha decidido unir ambas clases, y formar una clase conjunta de 923 ejemplares.

# %% [markdown]
# Creamos el dataset. Para ello, me ayudaré de fastai y crearé un conjunto de imágenes que se usarán para train y test. Nota: hay dos columnas de etiquetas, por lo que se han hecho dos funciones distintas; una que se ocupa de las etiquetas binarias, y otra de las subcategorías. Comenzaremos primero por clasificación binaria, teniendo cuidado de no introducir imágenes aumentadas en validación.

# %%
def mover_imgs(img_list, dir):
    os.mkdir(dir)
    for img in img_list:
        try:
            src = os.path.join("trainThumbnails", img)
            dst = os.path.join(dir, img)
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Error al copiar '{img}': {e}")

# %%
"""train_imgs_names = train_malignant['image'].to_list()
mover_imgs(train_imgs_names, 'malignantThumbnails')"""

# %%
def multi_label_malignant(fname):
    global train_malignant
    el = train_malignant.loc[TRAIN_DIR + train_malignant['image'] == str(fname)]
    return ((el['label'].values[0]))

# %%
class AlbumentationsTransform(DisplayedTransform):
    split_idx, order = 0, 2

    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# %%
def get_train_aug():
    return albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, constrast_limit= 0.2,p=0.7),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            #albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.8)
    ])

# %%
train_ds = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # La entrada es un bloque de imagenes, y la salida, categorias
    get_items=get_image_files,  # Utilizamos el mñetodo get_image_files proporcionado en el Notebook
    splitter=RandomSplitter(valid_pct=VALID_PCT, seed=SEED),
    get_y=multi_label_malignant,  # Las etiquetas son especificadas como contenido del dataset en si
    item_tfms=[Resize(IMAGE_RESIZE, method=RESIZE_METHOD), AlbumentationsTransform(get_train_aug())],  # Redimensionado
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders("malignantThumbnails/", bs=BATCH_SIZE)

# %% [markdown]
# Ahora, construimos un objeto dataloaders y creamos el objeto final apto para entrenamiento

# %%
learn_resnet = vision_learner(train_ds, 'resnet50', loss_func=LOSS_FUNC, metrics=METRICS)
print(learn_resnet.summary())

# %%
lrs = learn_resnet.lr_find()

# %%
# Creamos callbacks para el modelo
early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOP_PATIENCE)
save_callback = SaveModelCallback(monitor='valid_loss', fname='best_model_malignant')
callbacks = [early_stopping, save_callback, ShowGraphCallback]

# %% [markdown]
# Procedemos a realizar el finetuning. Para ello, primero se entrenará solo la head de la red, y posteriormente, la red en su conjunto. Se usará early stopping con patience=3

# %%
# FINETUNING
## Hiperparámetros
lr_mult = 100
base_lr = lrs[0] / 2

# Congelamos
learn_resnet.freeze()
learn_resnet.fit_one_cycle(FREEZE_EPOCHS, lrs[0], pct_start=0.99, cbs=callbacks)

# %%
perdidas = learn_resnet.recorder.plot_loss(skip_start=0, with_valid=True)
plt.savefig("perdidas50_freeze_malignant.png")

# %%
# Descongelamos y finalizamos entrenamiento
learn_resnet.unfreeze()
learn_resnet.fit_one_cycle(NUM_EPOCHS, slice(base_lr / lr_mult, base_lr), cbs=callbacks, wd=0.1)

# %%
perdidas = learn_resnet.recorder.plot_loss(skip_start=0, with_valid=True)
plt.savefig("perdidas50_unfreeze_malignant.png")

# %%
preds, targets = learn_resnet.get_preds(ds_idx=1)  # ds_idx=1 para el conjunto de validación

# %%
y_true = targets.numpy()
y_pred = preds.argmax(dim=1).numpy()  # Convierte las probabilidades en etiquetas predichas

# %%
print(train_ds.vocab)
print(classification_report(y_true, y_pred))

# %%
learn_resnet.save('modelo_fastai_bestISICmalignant')

# %% [markdown]
# ## Pytorch mobile
# 
# Convertimos a formato quantizado de pytorch mobile

# %%
# Convierte el modelo a TorchScript
model = learn_resnet.model.to('cpu')
model.eval()

scripted_module = torch.jit.script(nn.Sequential(model, nn.Softmax(dim=1)))
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
scripted_module.save("bestISICmalignant_512.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("bestISICmalignant 512_android.ptl")

# %%
# Sin optimizar
# Convierte el modelo a TorchScript
model = learn_resnet.model.to('cpu')
model.eval()

scripted_module = torch.jit.script(nn.Sequential(model, nn.Softmax(dim=1)))
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
scripted_module.save("bestISICmalignant_512_noop.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
scripted_module._save_for_lite_interpreter("bestISICmalignant 512_android_noop.ptl")


