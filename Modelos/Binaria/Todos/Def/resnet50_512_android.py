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
os.environ['MPLCONFIGDIR'] = '/mnt/homeGPU/hexecode/matplotlib/'

import numpy as np
import pandas as pd
import torch
import fastbook

import matplotlib.pyplot as plt
import seaborn as sns

from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fastai.vision.all import *
import torch.utils.mobile_optimizer as mobile_optimizer
from torch.utils.mobile_optimizer import optimize_for_mobile

"""
!pip install -Uqq fastbook
!pip install nbdev
"""
fastbook.setup_book()
torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

# %% [markdown]
# A continuación, se declaran las constantes usadas como parámetros a las funciones

# %%
print("Selected: ", torch.cuda.get_device_properties(torch.cuda.current_device()).name)
#torch.cuda.set_device(0)
#print("Selected after: ", torch.cuda.get_device_properties(torch.cuda.current_device()).name)
os.system("nvidia-smi")

# %%
# Número de clases. 2 para la binaria.
NUM_CLASSES = 2
SEED = 19
# Número de canales de cada imagen
CHANNELS = 3
IMAGE_RESIZE = 512
RESIZE_METHOD = 'squish'
LOSS_FUNC = CrossEntropyLossFlat()
METRICS = [accuracy, BalancedAccuracy(), error_rate, Recall(), Precision()]

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 120
FREEZE_EPOCHS = 120

EARLY_STOP_PATIENCE = 3

BATCH_SIZE = 32

train_aug = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
aug_df = pd.concat([train_aug, valid])

np.random.seed(SEED)

# %% [markdown]
# Ahora, podemos volver a calcular la proporción:

# %%
print(train_aug.bin.value_counts())

sns.countplot(data=train_aug, x='bin', order=train_aug.bin.value_counts().index)

# %%
"""
t = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
r = torch.cuda.memory_reserved(0) / (1024 ** 3)
a = torch.cuda.memory_allocated(0) / (1024 ** 3)
f = r - a  # free inside reserved
print("Memoria total: ", t, "GB")
print("Memoria reservada: ", r, "GB")
print("Memoria allocated: ", a, "GB")
print("Memoria libre: ", f, "GB")
"""

# %% [markdown]
# Creamos el dataset. Para ello, me ayudaré de fastai y crearé un conjunto de imágenes que se usarán para train y test. Nota: hay dos columnas de etiquetas, por lo que se han hecho dos funciones distintas; una que se ocupa de las etiquetas binarias, y otra de las subcategorías. Comenzaremos primero por clasificación binaria, teniendo cuidado de no introducir imágenes aumentadas en validación.

# %%
def binary_label(fname):
    global aug_df
    el = aug_df.loc[aug_df['image'] == str(fname).split("/")[2]]
    return el['bin'].values[0]


def multi_label(fname):
    global train_df
    el = train_df.loc[train_df['image'] == str(fname)]
    return ((el['label'].values[0]))


# %%
def val_splitter(fname):
    if Path(fname).parent.name == 'valid':
        return True
    else:
        return False


# %%
train_ds = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # La entrada es un bloque de imagenes, y la salida, categorias
    get_items=get_image_files,  # Utilizamos el mñetodo get_image_files proporcionado en el Notebook
    splitter=FuncSplitter(val_splitter),
    get_y=binary_label,  # Las etiquetas son especificadas como contenido del dataset en si
    item_tfms=[Resize(IMAGE_RESIZE, method=RESIZE_METHOD)],  # Redimensionado
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders("binAUG", bs=BATCH_SIZE)

# %% [markdown]
# Ahora, construimos un objeto dataloaders y creamos el objeto final apto para entrenamiento

# %%
learn_resnet = vision_learner(train_ds, 'resnet50', loss_func=LOSS_FUNC, metrics=METRICS)
print(learn_resnet.summary())

# %%
"""
t = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
r = torch.cuda.memory_reserved(0) / (1024 ** 3)
a = torch.cuda.memory_allocated(0) / (1024 ** 3)
f = r - a  # free inside reserved
print("Memoria total: ", t, "GB")
print("Memoria reservada: ", r, "GB")
print("Memoria allocated: ", a, "GB")
print("Memoria libre: ", f, "GB")
"""
# %%
# Creamos callbacks para el modelo
early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOP_PATIENCE)
save_callback = SaveModelCallback(monitor='valid_loss', fname='best_model_augmented')
callbacks = [early_stopping, save_callback, ShowGraphCallback]

# %% [markdown]
# Procedemos a realizar el finetuning. Para ello, primero se entrenará solo la head de la red, y posteriormente, la red en su conjunto. Se usará early stopping con patience=3

# %%
lrs = learn_resnet.lr_find()

os.system("nvidia-smi")

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
plt.savefig("perdidas_freeze.png")

# %%
# Descongelamos y finalizamos entrenamiento
learn_resnet.unfreeze()
learn_resnet.fit_one_cycle(NUM_EPOCHS, slice(base_lr / lr_mult, base_lr), cbs=callbacks)

# %%
perdidas = learn_resnet.recorder.plot_loss(skip_start=0, with_valid=True)
plt.savefig("perdidas_unfreeze.png")

# %%
state = torch.load('models/best_model_augmented.pth')
preds, targets = learn_resnet.get_preds(ds_idx=1)  # ds_idx=1 para el conjunto de validación

# %%
y_true = targets.numpy()
y_pred = preds.argmax(dim=1).numpy()  # Convierte las probabilidades en etiquetas predichas

# %%
print(classification_report(y_true, y_pred, target_names=train_ds.vocab))

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
scripted_module.save("skin-rn50-pc512.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("skin-rn50android512.ptl")
