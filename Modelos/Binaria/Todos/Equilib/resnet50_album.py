# %% [markdown]
# ## Entrenamiento de ResNet50 con data augmentation y 3 modelos
# 
# Se procede al particionado de datos de train y valid por separado para aplicar data augmentation en entrenamiento mediante la librería imgaug. Luego, se proceden a realizar un modelo de clasificación binaria para discriminar primero el tipo de enfermedad, y posteriormente, se creará un modelo específico para enfermedades cancerígenas y no cancerígenas.

# %% [markdown]
# Comenzamos por la importación de librerías

# %%
import os
os.environ['HF_HOME'] = '/mnt/homeGPU/hexecode/cache/'

import albumentations
# Librerías utilizadas por el script
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from fastai.vision.all import *
from fastai.callback.all import *


from sklearn.model_selection import train_test_split
import torch.utils.mobile_optimizer as mobile_optimizer
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils.data.sampler import WeightedRandomSampler


print("Selected: ", torch.cuda.get_device_properties(torch.cuda.current_device()).name)

# %%
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# %% [markdown]
# A continuación, se declaran las constantes usadas como parámetros a las funciones

# %%
# Número de clases. 2 para la binaria.
NUM_CLASSES = 2
TRAIN_DIR = "trainThumbnails/"
SEED = 19
set_seed(SEED)
# Número de canales de cada imagen
CHANNELS = 3
IMAGE_RESIZE = 460
VALID_PCT = 0.3
RESIZE_METHOD = 'squish'
LOSS_FUNC = FocalLoss()
METRICS = [accuracy, BalancedAccuracy(), error_rate, Recall(), Precision()]

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 100
FREEZE_EPOCHS = 80

EARLY_STOP_PATIENCE = 3

BATCH_SIZE = 32

np.random.seed(SEED)

# %% [markdown]
# Realizamos la seperación de entrenamiento y validación, junto el aumento de datos. Primero, se separa el entrenamiento para evitar un sesgo de los resultados en validación, y posteriormente, se procede a la construcción de los dataframes.

# %%
train_df = pd.read_csv('trainSet.csv')
#display(train_df)

# %%
# Separación validación y train

train, valid = train_test_split(train_df, test_size=VALID_PCT, stratify=train_df[['bin']])

# Vemos cómo es el desequilibrio entre clases
print(train.bin.value_counts())

sns.countplot(data=train, x='bin', order=train_df.bin.value_counts().index)



# %%
 # Hallamos cuál es la proporción entre las dos clases para igualarlas mediante aumento de datos
benign_count, malign_count = train.bin.value_counts()
print("MAL/BENG: ", benign_count / malign_count)

# %% [markdown]
# Podemos generar hasta 3 imágenes por data augmentation de la clase maligna para equiparar aproximadamente ambas clases

# %%
def binary_label(fname):
    global train_df
    el = train_df.loc[train_df['image'] == str(fname).split("/")[1]]
    return el['bin'].values[0]

# %%
def get_train_aug():
    return albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(IMAGE_RESIZE, IMAGE_RESIZE),
        #albumentations.Normalize()
    ])


def get_valid_aug():
    return albumentations.Compose([
        albumentations.Resize(IMAGE_RESIZE, IMAGE_RESIZE),
        #albumentations.Normalize()
    ])


# %%
class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# %%
item_tfms = [Resize(IMAGE_RESIZE), AlbumentationsTransform(get_train_aug())]

# %%
train_ds = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # La entrada es un bloque de imagenes, y la salida, categorias
    get_items=get_image_files,  # Utilizamos el metodo get_image_files proporcionado en el Notebook
    splitter=RandomSplitter(valid_pct=VALID_PCT, seed=SEED),
    get_y=binary_label,  # Las etiquetas son especificadas como contenido del dataset en si
    item_tfms=item_tfms,  # Redimensionado
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders(TRAIN_DIR, bs=BATCH_SIZE)

# %%
train_ds.train.show_batch()

# %%
learn_resnet = vision_learner(train_ds, 'resnet50', loss_func=LOSS_FUNC, metrics=METRICS, pretrained=True)
print(learn_resnet.summary())

# %%
lrs = learn_resnet.lr_find()

# Creamos callbacks para el modelo
early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOP_PATIENCE)
save_callback = SaveModelCallback(monitor='valid_loss', fname='best_model_augmented_freeze32')
callbacks = [early_stopping, save_callback, ShowGraphCallback]

# %%
lr_mult = 100
base_lr = lrs[0] / 2
print("ENTRENAMIENTO: Finetuning")
print("=========================\n Fase de congelado:")
print(f" LR: {lrs[0]}")

# %%
# Congelamos
learn_resnet.freeze()
learn_resnet.fit_one_cycle(FREEZE_EPOCHS, lrs[0], pct_start=0.99, cbs=callbacks)

# %%
perdidas = learn_resnet.recorder.plot_loss(skip_start=0, with_valid=True)
plt.savefig("perdidas50_freeze_album.png")

# %%
# Descongelamos y finalizamos entrenamiento

early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOP_PATIENCE)
save_callback = SaveModelCallback(monitor='valid_loss', fname='best_model_augmented32')
callbacks = [early_stopping, save_callback, ShowGraphCallback]

print("=========================\n Fase de descongelado:")
print(f" LR: {base_lr}")

learn_resnet.unfreeze()
learn_resnet.fit_one_cycle(NUM_EPOCHS, slice(base_lr / lr_mult, base_lr), cbs=callbacks)

# %%
perdidas = learn_resnet.recorder.plot_loss(skip_start=0, with_valid=True)
plt.savefig("perdidas50_unfreeze_album.png")

# %%
preds, targets = learn_resnet.get_preds(ds_idx=1)  # ds_idx=1 para el conjunto de validación

# %%
y_true = targets.numpy()
y_pred = preds.argmax(dim=1).numpy()  # Convierte las probabilidades en etiquetas predichas

# %%
print(classification_report(y_true, y_pred, target_names=train_ds.vocab))

# %%
# Convierte el modelo a TorchScript
model = learn_resnet.model.to('cpu')
model.eval()

scripted_module = torch.jit.script(nn.Sequential(model, nn.Softmax(dim=1)))
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
scripted_module.save("skin-rn50-album.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("skin-rn50android460.ptl")


