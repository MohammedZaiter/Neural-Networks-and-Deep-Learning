"""Helper functions for SafeSight_PPE_Deep_Learning.ipynb.

The notebook keeps the explanations, the experiment definitions and the observations.
Everything reusable lives here:

    0. Paths, constants and configuration
    1. Dataset: location, annotation audit, cropping, duplicate check
    2. Input pipeline: crop cache, tf.data datasets, class weights
    3. Models: MLP, CNN from scratch, MobileNetV2 transfer
    4. Training: optimizer, callbacks, run() for one experiment
    5. Metrics and benchmark tables
    6. Plots
"""
import json
import hashlib
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -----------------------------------------------------
# 0. Paths, constants and configuration
# -----------------------------------------------------
SEED = 504

IN_COLAB = 'google.colab' in sys.modules

# Paths are relative to the Notebooks/ folder. On Colab the dataset and the crop cache stay on the
# local disk (fast); checkpoints and results are stored in the project folder on Drive so they are not lost.
DATA_DIR = Path('/content/data') if IN_COLAB else Path('../data')
CKPT = Path('../Checkpoints')          # best checkpoint of every experiment + the final model
RESULTS = Path('../results')           # CSV / JSON results
FIGS = RESULTS / 'figures'             # plots

# Roboflow API key used to download the dataset when it is not in DATA_DIR. An environment variable or a
# Colab secret named ROBOFLOW_API_KEY takes priority over this value.
ROBOFLOW_API_KEY = 'rf_nNt129Th3GhJM5OKn1tyH4sLPGK2'

TARGET = ['Hardhat', 'NO-Hardhat', 'Safety Vest', 'NO-Safety Vest']
UNSAFE = (1, 3)                        # NO-Hardhat, NO-Safety Vest

CONFIG = {'workspace': 'roboflow-universe-projects', 'project': 'construction-site-safety', 'version': 27,
          'image_size': 160,       # crop resolution fed to every model
          'crop_padding': 0.10,    # extra context around each box, as a fraction of box size
          'min_box_pixels': 12,    # outlier rule: drop crops smaller than this after clipping
          'batch_size': 32,
          'scratch_epochs': 15, 'transfer_epochs': 15, 'fine_tune_epochs': 8,
          'ablation_epochs': 15,     # 4.1-4.2: same budget as the baseline CNN (3.2)
          'long_epochs': 30,         # 4.1: number-of-epochs experiment, without early stopping
          'best_cnn_epochs': 30,     # 4.4: the combined CNN gets a longer budget
          'tl_compare_epochs': 10,   # 5.4: same budget for every transfer setup
          'small_fraction': 0.10}    # 5.4: share of training images in the small-data runs

# Training settings of the baseline CNN (3.2); the controlled experiments override them one at a time
BASE_TRAIN = dict(lr=1e-3, optimizer='adam', batch=32, schedule='plateau', class_weights=True,
                  loss='ce', early_stopping=True)

DEVICE = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
RETRAIN = False        # set by setup()


def set_seed(seed=SEED):
    """One seed for Python, NumPy and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def setup(retrain=False):
    """Seed everything, create the output folders, save the configuration and reload earlier results.
    retrain=False: experiments with a saved checkpoint are loaded; only missing ones are trained.
    retrain=True: every experiment is trained again and its checkpoint overwritten."""
    global RETRAIN
    RETRAIN = retrain
    set_seed()
    if DEVICE == 'CPU':
        # Bit-exact reproducibility on CPU. Some GPU kernels have no deterministic version, so GPU runs are
        # seeded but can differ slightly from run to run.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    for folder in (DATA_DIR, CKPT, RESULTS, FIGS):
        folder.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'configuration.json').write_text(json.dumps(CONFIG, indent=2))
    _load_saved_results()
    print('TensorFlow:', tf.__version__, '| device:', DEVICE, '| retrain:', RETRAIN)


# -----------------------------------------------------
# 1. Dataset: location, annotation audit, cropping, duplicate check
# -----------------------------------------------------
def find_dataset():
    """Dataset folder and source class names. Priority: the LOCAL_DATASET_DIR environment variable,
    then data/Construction-Site-Safety-27/, then a download from Roboflow with ROBOFLOW_API_KEY
    (environment variable or Colab secret)."""
    local = os.environ.get('LOCAL_DATASET_DIR', '').strip()
    default = DATA_DIR / f"Construction-Site-Safety-{CONFIG['version']}"
    if local:
        data = Path(local).expanduser().resolve()
    elif (default / 'data.yaml').exists():
        data = default.resolve()
    else:
        key = os.environ.get('ROBOFLOW_API_KEY', '').strip()
        if not key:
            try:
                from google.colab import userdata
                key = userdata.get('ROBOFLOW_API_KEY') or ''
            except Exception:
                pass
        if not key:
            key = ROBOFLOW_API_KEY  # project key, used when no environment variable or Colab secret is set
        if not key:
            raise RuntimeError('Set ROBOFLOW_API_KEY (env/Colab secret), place the YOLOv8 export in '
                               f'{default}, or set LOCAL_DATASET_DIR.')
        from roboflow import Roboflow
        version = Roboflow(api_key=key).workspace(CONFIG['workspace']).project(CONFIG['project']).version(CONFIG['version'])
        data = Path(version.download('yolov8', location=str(default)).location).resolve()

    assert data.exists(), f'Missing dataset folder: {data}'
    yaml_path = next((p for p in [data / 'data.yaml', data / 'dataset.yaml'] if p.exists()), None)
    assert yaml_path is not None, 'Expected data.yaml or dataset.yaml at export root'
    raw = yaml.safe_load(yaml_path.read_text())['names']
    # data.yaml stores class names either as a list or as {id: name}
    names = {int(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else dict(enumerate(raw))
    return data, names


def canon(s):
    """Normalize a class name so that e.g. 'NO-Safety Vest' and 'no_safety_vest' match."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _split_folders(data, split):
    """(images, labels) folders of a split; Roboflow may call validation 'valid' or 'val'."""
    for k in (['valid', 'val'] if split == 'valid' else [split]):
        if (data / k / 'images').exists() and (data / k / 'labels').exists():
            return data / k / 'images', data / k / 'labels'
    raise FileNotFoundError(f'Missing {split} images/labels under {data}')


def audit_and_crop(data, names):
    """Check every image and label row, apply the outlier rule and list one crop per valid box.
    Returns DF (one row per crop) and AUD (one row per rejected image or annotation, with the reason)."""
    # Map the source classes onto the 4 target labels; all other classes are ignored
    lookup = {canon(s): i for i, s in enumerate(TARGET)}
    source_to_label = {i: lookup[canon(s)] for i, s in names.items() if canon(s) in lookup}
    print('Source label mapping:', {names[i]: TARGET[j] for i, j in source_to_label.items()})
    if set(source_to_label.values()) != set(range(4)):
        raise ValueError('Selected export does not contain all four target classes; inspect data.yaml or use another version.')

    records, audits, parent_sets = [], [], {}
    pad = CONFIG['crop_padding']
    for split in ['train', 'valid', 'test']:
        image_dir, label_dir = _split_folders(data, split)
        ids = set()
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        for imgpath in images:
            ids.add(imgpath.stem)
            labelpath = label_dir / (imgpath.stem + '.txt')
            im = cv2.imread(str(imgpath))
            if im is None:
                audits.append((split, imgpath.name, 'unreadable image'))
                continue
            h, w = im.shape[:2]
            if not labelpath.exists():
                audits.append((split, imgpath.name, 'missing annotation'))
                continue
            for rownum, line in enumerate(labelpath.read_text().splitlines()):
                # Missing values / malformed rows are audited and skipped, never imputed
                tokens = line.split()
                if len(tokens) != 5:
                    audits.append((split, imgpath.name, 'invalid label row'))
                    continue
                try:
                    cid = int(tokens[0])
                    xc, yc, bw, bh = map(float, tokens[1:])
                except (ValueError, TypeError):
                    audits.append((split, imgpath.name, 'non-numeric annotation'))
                    continue
                if cid not in source_to_label:
                    continue  # not one of the 4 target classes
                if not np.all(np.isfinite([xc, yc, bw, bh])) or bw <= 0 or bh <= 0 or not (0 <= xc <= 1 and 0 <= yc <= 1):
                    audits.append((split, imgpath.name, 'invalid coordinates'))
                    continue
                # Normalized YOLO box -> padded pixel coordinates, clipped to the image
                x1 = max(int((xc - bw * (.5 + pad)) * w), 0)
                x2 = min(int((xc + bw * (.5 + pad)) * w), w)
                y1 = max(int((yc - bh * (.5 + pad)) * h), 0)
                y2 = min(int((yc + bh * (.5 + pad)) * h), h)
                # Outlier rule: boxes too small to show recognizable equipment
                if min(x2 - x1, y2 - y1) < CONFIG['min_box_pixels']:
                    audits.append((split, imgpath.name, 'small clipped box'))
                    continue
                records.append(dict(split=split, path=str(imgpath), parent=imgpath.name, annotation=rownum,
                                    label=source_to_label[cid], x1=x1, y1=y1, x2=x2, y2=y2,
                                    width=x2 - x1, height=y2 - y1, area=(x2 - x1) * (y2 - y1)))
        parent_sets[split] = ids

    df = pd.DataFrame(records)
    aud = pd.DataFrame(audits, columns=['split', 'image', 'reason'])

    # Sanity checks before any modeling
    assert not df.empty, 'No eligible crops. Confirm dataset version and annotation classes.'
    assert not (parent_sets['train'] & parent_sets['valid'] or parent_sets['train'] & parent_sets['test']
                or parent_sets['valid'] & parent_sets['test']), 'Same filename across image splits: investigate leakage.'
    assert all(set(df.loc[df.split == s, 'label']) == set(range(4)) for s in ['train', 'valid', 'test']), \
        'A partition is missing a target category; investigate before proceeding.'

    # Saved manifest uses paths relative to the dataset folder (no machine-specific absolute paths)
    df.assign(path=df.path.map(lambda p: Path(p).relative_to(data).as_posix())).to_csv(RESULTS / 'crop_manifest.csv', index=False)
    aud.to_csv(RESULTS / 'annotation_audit.csv', index=False)
    return df, aud


def duplicate_check(df):
    """Exact-duplicate check: SHA-256 of every parent image, compared across splits."""
    hashes = {sp: {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df[df.split == sp].path.drop_duplicates()}
              for sp in ('train', 'valid', 'test')}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlap = hashes[a] & hashes[b]
        print(f'{a}/{b} exact-image hash overlaps: {len(overlap)}')
        if overlap:
            raise RuntimeError('Cross-split exact duplicates detected: fix dataset export before fitting models.')
    print('Note: augmented/near-duplicate detection also requires perceptual review beyond SHA256.')


def read_crop(row):
    """One crop as an RGB array at its original size (for display)."""
    im = cv2.imread(row.path)
    return cv2.cvtColor(im[row.y1:row.y2, row.x1:row.x2], cv2.COLOR_BGR2RGB)


# -----------------------------------------------------
# 2. Input pipeline: crop cache, tf.data datasets, class weights
# -----------------------------------------------------
DF = None              # crop manifest, set by prepare_data()
CROPS = None           # every crop as uint8, S x S x 3, in DF order
TRAIN_ROWS = None      # DF indices of the training crops
CLASS_WEIGHT = None    # class weights of the full training set
TRAIN_DS = VAL_DS = None   # default training (shuffled) and validation datasets


def _cut_crops(df, size):
    crops = np.empty((len(df), size, size, 3), np.uint8)
    for path, rows in df.groupby('path', sort=False):    # decode each parent image only once
        image = cv2.imread(path)
        if image is None:
            raise OSError(path)
        for i, r in rows.iterrows():
            crop = cv2.cvtColor(image[r.y1:r.y2, r.x1:r.x2], cv2.COLOR_BGR2RGB)
            crops[i] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crops


def prepare_data(df):
    """Cut every crop once (BGR -> RGB, resize to S x S, uint8), cache it in data/ and keep it in memory.
    Also stores the training indices and the class weights of the full training set."""
    global DF, CROPS, TRAIN_ROWS, CLASS_WEIGHT, TRAIN_DS, VAL_DS
    S = CONFIG['image_size']
    DF = df
    # The cache file name records every setting that changes the crops
    cache = DATA_DIR / f"crops_v{CONFIG['version']}_{S}px_pad{CONFIG['crop_padding']}_min{CONFIG['min_box_pixels']}.npy"
    if cache.exists() and np.load(cache, mmap_mode='r').shape[0] == len(df):
        CROPS = np.load(cache)
    else:
        CROPS = _cut_crops(df, S)
        np.save(cache, CROPS)
    TRAIN_ROWS = df.index[df.split == 'train'].to_numpy()
    CLASS_WEIGHT = class_weights_for(TRAIN_ROWS)
    TRAIN_DS, VAL_DS = make_ds('train', True), make_ds('valid')
    print(f'{len(CROPS):,} crops in memory ({CROPS.nbytes / 1e9:.2f} GB)')


def make_ds(split, shuffle=False, batch=None, rows=None):
    """Batches of (image in [0, 1], label) for one split. `rows` restricts it to a subset of crops.
    Only training data is shuffled."""
    S = CONFIG['image_size']
    idx = DF.index[DF.split == split].to_numpy() if rows is None else np.asarray(rows)
    ds = tf.data.Dataset.from_tensor_slices((idx, DF.label.to_numpy()[idx]))
    if shuffle:
        ds = ds.shuffle(len(idx), seed=SEED, reshuffle_each_iteration=True)

    def fetch(i, label):
        images = tf.numpy_function(lambda k: CROPS[k], [i], tf.uint8)
        images.set_shape((None, S, S, 3))
        return tf.cast(images, tf.float32) / 255., tf.cast(label, tf.int32)

    return ds.batch(batch or CONFIG['batch_size']).map(fetch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def class_weights_for(rows):
    """Inverse class frequency over the given training crops, so rarer classes count more in the loss."""
    counts = np.bincount(DF.label.to_numpy()[rows], minlength=4)
    return {i: float(len(rows) / (4 * max(n, 1))) for i, n in enumerate(counts)}


def small_subset(fraction, seed=SEED):
    """Training crops of a random `fraction` of the training images, drawn by parent image
    so that crops of one image stay together. Returns (DF indices, number of images, total images)."""
    parents = np.sort(DF.loc[DF.split == 'train', 'parent'].unique())
    chosen = np.random.default_rng(seed).choice(parents, int(round(fraction * len(parents))), replace=False)
    rows = DF.index[(DF.split == 'train') & DF.parent.isin(chosen)].to_numpy()
    assert set(DF.label[rows]) == set(range(4)), 'The small subset is missing a class: change the seed or fraction.'
    return rows, len(chosen), len(parents)


# -----------------------------------------------------
# 3. Models: MLP, CNN from scratch, MobileNetV2 transfer
# -----------------------------------------------------
def aug():
    """Training-time augmentation; Keras disables these layers automatically at inference."""
    return keras.Sequential([layers.RandomFlip('horizontal'),
                             layers.RandomRotation(.07),
                             layers.RandomContrast(.12)], name='train_augmentation')


def make_mlp():
    """Resizing(64 x 64) -> Flatten -> Dense(128, ReLU) -> Dropout(0.35) -> Dense(4, softmax)."""
    S = CONFIG['image_size']
    i = keras.Input((S, S, 3))
    x = layers.Resizing(64, 64)(i)          # 64 x 64 x 3 = 12,288 inputs keeps the dense layer manageable
    x = layers.Flatten()(x)                 # spatial structure is discarded here
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(.35)(x)
    return keras.Model(i, layers.Dense(4, activation='softmax')(x), name='baseline_mlp')


def make_cnn(activation='relu', filters=24, dropout=.3, l2=1e-4, augmentation=True,
             blocks=3, kernel=3, downsample='pool', batchnorm=True, head='gap'):
    """CNN from scratch. The defaults build the baseline: three Conv(3x3) -> BatchNorm -> activation -> MaxPool
    blocks with 24, 48 and 96 filters, then GlobalAveragePooling -> Dropout -> Dense(4, softmax).
    downsample: 'pool' (2x2 max-pooling) or 'stride' (stride-2 convolution, no pooling).
    head: 'gap' (global average pooling) or 'flatten'."""
    S = CONFIG['image_size']
    i = keras.Input((S, S, 3))
    x = aug()(i) if augmentation else i
    for b in range(blocks):                      # the number of filters doubles in every block
        x = layers.Conv2D(filters * 2 ** b, kernel, strides=2 if downsample == 'stride' else 1, padding='same',
                          kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if downsample == 'pool':
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x) if head == 'gap' else layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    return keras.Model(i, layers.Dense(4, activation='softmax')(x), name=f'cnn_{activation}_{filters}')


def build_transfer():
    """augmentation -> rescale to [-1, 1] -> MobileNetV2 (ImageNet, frozen) -> GAP -> Dense(128, ReLU)
    -> Dropout(0.4) -> Dense(4, softmax). Returns (model, backbone)."""
    S = CONFIG['image_size']
    inp = keras.Input(shape=(S, S, 3))
    x = aug()(inp)
    x = layers.Rescaling(2., offset=-1.)(x)   # [0, 1] -> [-1, 1], the input range MobileNetV2 was trained on
    backbone = keras.applications.MobileNetV2(input_shape=(S, S, 3), include_top=False, weights='imagenet')
    backbone.trainable = False                # pretrained features stay fixed
    x = backbone(x, training=False)           # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(.4)(x)
    return keras.Model(inp, layers.Dense(4, activation='softmax')(x), name='mobilenetv2_ppe'), backbone


def get_backbone(model):
    """The MobileNetV2 backbone nested inside a transfer model."""
    return next(l for l in model.layers if isinstance(l, keras.Model) and l.name.startswith('mobilenetv2'))


def unfreeze(backbone, last=None):
    """Make the last `last` backbone layers trainable (all of them when last is None).
    BatchNorm layers always stay frozen, so small batches cannot overwrite their ImageNet statistics."""
    backbone.trainable = True
    n = len(backbone.layers)
    for k, layer in enumerate(backbone.layers):
        layer.trainable = (not isinstance(layer, layers.BatchNormalization)) and (last is None or k >= n - last)


def make_transfer(mode):
    """MobileNetV2 with a new head. mode: 'frozen' (head only), 'last30' (last 30 backbone layers + head)
    or 'all' (whole network, BatchNorm frozen)."""
    model, backbone = build_transfer()
    if mode != 'frozen':
        unfreeze(backbone, 30 if mode == 'last30' else None)
    return model


# -----------------------------------------------------
# 4. Training: optimizer, callbacks, run() for one experiment
# -----------------------------------------------------
VAL_ROWS = []     # one row of validation metrics per experiment
BENCH_ROWS = []   # one benchmark row per experiment
HISTORIES = {}    # Keras training history per experiment
_SAVED = {'times': {}, 'histories': {}, 'bench': {}}


def _load_saved_results():
    """Results of earlier runs, reused when a model is loaded instead of retrained."""
    if (RESULTS / 'validation_results.csv').exists():
        _SAVED['times'] = pd.read_csv(RESULTS / 'validation_results.csv').set_index('model')['training_seconds'].to_dict()
    if (RESULTS / 'training_histories.json').exists():
        _SAVED['histories'] = json.loads((RESULTS / 'training_histories.json').read_text())
        HISTORIES.update(_SAVED['histories'])
    if (RESULTS / 'experiment_benchmark.csv').exists():
        _SAVED['bench'] = pd.read_csv(RESULTS / 'experiment_benchmark.csv').set_index('model').to_dict('index')


@tf.keras.utils.register_keras_serializable(package='safesight')
class SparseFocalLoss(keras.losses.Loss):
    """Focal loss (Lin et al., 2017) for integer labels: FL = -(1 - p_t)^gamma * log(p_t), where p_t is the
    predicted probability of the true class. Easy, well-classified crops (p_t close to 1) get a small loss,
    so training focuses on the hard ones. gamma = 0 gives the usual cross-entropy."""

    def __init__(self, gamma=2.0, name='sparse_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        p_t = tf.clip_by_value(tf.gather(y_pred, y_true, axis=1, batch_dims=1), 1e-7, 1.0)
        return -tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

    def get_config(self):
        return {**super().get_config(), 'gamma': self.gamma}


def compile_model(model, lr=1e-3, optimizer='adam', decay_steps=None, loss='ce'):
    """Adam or SGD with momentum 0.9. With decay_steps, the learning rate follows a cosine decay.
    loss: 'ce' (sparse categorical cross-entropy) or 'focal' (SparseFocalLoss, gamma 2)."""
    rate = keras.optimizers.schedules.CosineDecay(lr, decay_steps) if decay_steps else lr
    opt = keras.optimizers.Adam(learning_rate=rate) if optimizer == 'adam' else keras.optimizers.SGD(learning_rate=rate, momentum=.9)
    model.compile(optimizer=opt, loss=SparseFocalLoss() if loss == 'focal' else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def callbacks(name, schedule='plateau', early_stopping=True):
    cbs = [
        # save the best checkpoint of this experiment
        keras.callbacks.ModelCheckpoint(str(CKPT / f'{name}.keras'), monitor='val_loss', save_best_only=True),
    ]
    if early_stopping:
        # stop when validation loss has not improved for 4 epochs, keep the best weights
        cbs.insert(0, keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True))
    if schedule == 'plateau':
        # halve the learning rate when validation loss plateaus for 2 epochs
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2, min_lr=1e-6))
    return cbs


def run(name, model, lr, epochs, train=None, validation=None, optimizer='adam', schedule='plateau',
        weights='auto', train_rows=None, group='', change='', loss='ce', early_stopping=True):
    """Train one experiment (or load its checkpoint when RETRAIN is False), score it on validation
    and log its metrics, training history and benchmark row to results/.
    schedule: 'plateau' (ReduceLROnPlateau), 'cosine' (cosine decay) or 'none'.
    weights: class weights; 'auto' uses those of the full training set, None disables them.
    loss: 'ce' (cross-entropy) or 'focal'. early_stopping=False trains for all the epochs."""
    train = TRAIN_DS if train is None else train
    validation = VAL_DS if validation is None else validation
    weights = CLASS_WEIGHT if isinstance(weights, str) else weights
    rows = TRAIN_ROWS if train_rows is None else train_rows
    ckpt = CKPT / f'{name}.keras'
    trained = RETRAIN or not ckpt.exists()
    if not trained:
        # Reuse the trained model instead of training it again
        model = load(name)
        print(f'{name}: loaded trained model from {ckpt.as_posix()}')
        seconds = _SAVED['times'].get(name, float('nan'))
        if name in _SAVED['histories']:
            HISTORIES[name] = _SAVED['histories'][name]
    else:
        compile_model(model, lr, optimizer, len(train) * epochs if schedule == 'cosine' else None, loss)
        t = time.time()
        h = model.fit(train, validation_data=validation, epochs=epochs,
                      callbacks=callbacks(name, schedule, early_stopping), class_weight=weights, verbose=2)
        seconds = round(time.time() - t, 2)
        if ckpt.exists():
            model.load_weights(ckpt)   # evaluate exactly the saved best epoch
        else:
            model.save(ckpt)           # validation loss was never finite (training diverged)
        HISTORIES[name] = {m: [float(v) for v in vals] for m, vals in h.history.items()}
        (RESULTS / 'training_histories.json').write_text(json.dumps({**_SAVED['histories'], **HISTORIES}, indent=1))
    metrics = scores(model, validation, 'valid')
    row = {'model': name, 'training_seconds': seconds, **metrics}
    VAL_ROWS.append(row)
    _save_rows(VAL_ROWS, RESULTS / 'validation_results.csv')
    BENCH_ROWS.append(benchmark(name, model, metrics, seconds, rows, trained, group, change))
    _save_rows(BENCH_ROWS, RESULTS / 'experiment_benchmark.csv')
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return model


def train_cnn(name, model_kw, train_kw, epochs, seed=SEED, rows=None, group='', change=''):
    """Build make_cnn(**model_kw) and train it with BASE_TRAIN updated by train_kw (via run())."""
    tk = {**BASE_TRAIN, **train_kw}
    rows = TRAIN_ROWS if rows is None else rows
    set_seed(seed)
    return run(name, make_cnn(**model_kw), tk['lr'], epochs,
               train=make_ds('train', True, tk['batch'], rows), validation=make_ds('valid', False, tk['batch']),
               optimizer=tk['optimizer'], schedule=tk['schedule'],
               weights=class_weights_for(rows) if tk['class_weights'] else None,
               train_rows=rows, group=group, change=change, loss=tk['loss'], early_stopping=tk['early_stopping'])


def load(name):
    """Load the best checkpoint of an experiment (for prediction; run() compiles again before any training)."""
    return keras.models.load_model(CKPT / f'{name}.keras', compile=False)


def clear():
    """Free the memory of finished models between experiments."""
    keras.backend.clear_session()


# -----------------------------------------------------
# 5. Metrics and benchmark tables
# -----------------------------------------------------
def scores(model, ds, sp='valid'):
    """Accuracy, macro-F1, balanced accuracy and mean recall of the unsafe classes (NO-Hardhat, NO-Safety Vest)."""
    y = DF[DF.split == sp].label.to_numpy()
    pred = model.predict(ds, verbose=0).argmax(axis=1)
    return dict(accuracy=float(np.mean(pred == y)),
                macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                unsafe_recall=float(np.mean([np.mean(pred[y == i] == i) if (y == i).sum() else np.nan for i in UNSAFE])))


def count_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def receptive_field(model):
    """Receptive field, in input pixels, of one value of the last feature map:
    for every conv / pool layer, rf += (k - 1) * jump and jump *= stride.
    Not computed (NaN) for models without convolutions or with a nested pretrained backbone."""
    rf, jump, found = 1, 1, False
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            if any(isinstance(l, layers.Conv2D) for l in layer.layers):
                return float('nan')        # pretrained backbone: not computed
            continue                       # augmentation block
        if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
            k = layer.kernel_size[0] if isinstance(layer, layers.Conv2D) else layer.pool_size[0]
            rf += (k - 1) * jump
            jump *= layer.strides[0]
            found = True
    return rf if found else float('nan')


def latency_ms(model, n=50):
    """Average time to classify one image (batch of 1) with the compiled model, after a warm-up."""
    S = CONFIG['image_size']
    x = tf.zeros((1, S, S, 3))
    predict = tf.function(lambda inp: model(inp, training=False))
    for _ in range(5):
        predict(x).numpy()
    t = time.perf_counter()
    for _ in range(n):
        predict(x).numpy()
    return (time.perf_counter() - t) / n * 1000


def train_accuracy(model, rows):
    """Accuracy on the training crops in inference mode (no augmentation, no dropout)."""
    pred = model.predict(make_ds('train', rows=rows), verbose=0).argmax(1)
    return float(np.mean(pred == DF.label.to_numpy()[rows]))


def benchmark(name, model, metrics, seconds, rows, trained, group, change):
    """Same measurements for every experiment, as in the CIFAR-10 project. The generalization gap is training
    accuracy without augmentation or dropout minus validation accuracy (large and positive = over-fitting)."""
    saved = {} if trained else _SAVED['bench'].get(name, {})
    h = HISTORIES.get(name)
    epochs_run = best_epoch = peak = float('nan')
    if h:
        epochs_run = len(h['loss'])
        val_loss = np.array(h['val_loss'], dtype=float)
        best_epoch = int(np.nanargmin(val_loss)) + 1 if np.isfinite(val_loss).any() else float('nan')
        peak = float(np.max(h['val_accuracy']))
    train_acc = saved.get('train_acc', float('nan'))
    if pd.isna(train_acc):
        train_acc = train_accuracy(model, rows)
    return dict(model=name, group=group or saved.get('group', ''), change=change or saved.get('change', ''),
                device=DEVICE if trained else saved.get('device', 'CPU (first run)'),
                parameters=int(model.count_params()), trainable_parameters=count_trainable(model),
                receptive_field=receptive_field(model), latency_ms=latency_ms(model),
                epochs_run=epochs_run, best_epoch=best_epoch,
                seconds_per_epoch=seconds / epochs_run if epochs_run == epochs_run else float('nan'),
                peak_val_acc=peak, train_acc=train_acc, val_acc=metrics['accuracy'],
                gap=train_acc - metrics['accuracy'], val_macro_f1=metrics['macro_f1'],
                unsafe_recall=metrics['unsafe_recall'])


def _save_rows(rows, path):
    """Write rows to CSV, keeping the rows of experiments that were not run in this session."""
    new = pd.DataFrame(rows).drop_duplicates('model', keep='last')
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
    new.to_csv(path, index=False)


def bench_table():
    """Benchmark rows of every experiment run so far, one per experiment (read from results/)."""
    return pd.read_csv(RESULTS / 'experiment_benchmark.csv').set_index('model')


def results_table():
    """Validation metrics of every experiment run so far, ranked by macro-F1 (tie-break: unsafe-class recall)."""
    return pd.read_csv(RESULTS / 'validation_results.csv').sort_values(['macro_f1', 'unsafe_recall'], ascending=False)


def load_best_cnn_config():
    """Settings of the combined best CNN chosen in 4.4: (model settings, training settings)."""
    cfg = json.loads((RESULTS / 'best_cnn_config.json').read_text())
    return cfg['model'], {k: v for k, v in cfg['train'].items() if BASE_TRAIN.get(k) != v}


def layer_table(model):
    """Output shape and parameter count of every layer (input and augmentation layers excluded)."""
    return pd.DataFrame([dict(layer=l.name, type=type(l).__name__,
                              output_shape=' x '.join(str(d) for d in l.output.shape[1:]),
                              parameters=l.count_params())
                         for l in model.layers if not isinstance(l, (layers.InputLayer, keras.Sequential))])


def ablation_table(ablations):
    """Benchmark rows of the controlled experiments, each compared with the mean of the baseline seeds.
    A change is 'better' or 'worse' only when its macro-F1 lies outside the range of the baseline seeds.
    Returns (table, baseline mean, baseline min, baseline max)."""
    abl = bench_table().loc[list(ablations)].copy()
    base = abl[abl.group == 'baseline']
    ref, lo, hi = base.val_macro_f1.mean(), base.val_macro_f1.min(), base.val_macro_f1.max()
    abl['delta'] = abl.val_macro_f1 - ref
    abl['effect'] = np.select([abl.val_macro_f1 > hi, abl.val_macro_f1 < lo], ['better', 'worse'], 'within noise')
    abl.loc[base.index, 'effect'] = 'baseline'
    return abl, ref, lo, hi


def combine_winners(abl, ablations):
    """Model and training settings of every change that beat the baseline beyond the seed range.
    When two winners set the same option, the better one is kept. If none beats the noise,
    every change with a positive mean difference is used instead.
    Returns (model settings, training settings, list of changes)."""
    winners = abl[abl.effect == 'better'].sort_values('val_macro_f1', ascending=False)
    if winners.empty:
        print('No single change beat the seed range; using every change with a positive mean difference instead.')
        winners = abl[(abl.group != 'baseline') & (abl.delta > 0)].sort_values('val_macro_f1', ascending=False)
    model_kw, train_kw, changes = {}, {}, []
    for name in winners.index:
        _, change, mk, tk = ablations[name]
        if set(mk) & set(model_kw) or set(tk) & set(train_kw):
            continue                                   # same option already set by a better change
        model_kw.update(mk)
        train_kw.update(tk)
        changes.append(change)
    return model_kw, train_kw, changes


def epochs_to(name, threshold):
    """First epoch whose validation accuracy reaches the threshold (NaN if never): convergence speed."""
    v = np.array(HISTORIES.get(name, {}).get('val_accuracy', []))
    hit = np.flatnonzero(v >= threshold)
    return int(hit[0]) + 1 if len(hit) else np.nan


def transfer_table(runs, setup_order):
    """Benchmark of the transfer comparison runs ({name: (setup, % of data, lr)}) plus convergence speed,
    and the data-efficiency table (validation macro-F1 with 10% vs 100% of the training images)."""
    tlc = bench_table().loc[list(runs)].copy()
    tlc['setup'] = [runs[n][0] for n in tlc.index]
    tlc['data'] = [f'{runs[n][1]}%' for n in tlc.index]
    tlc['lr'] = [runs[n][2] for n in tlc.index]
    tlc['epochs_to_70'] = [epochs_to(n, .70) for n in tlc.index]
    tlc['epochs_to_80'] = [epochs_to(n, .80) for n in tlc.index]
    eff = tlc.pivot(index='setup', columns='data', values='val_macro_f1').reindex(setup_order)
    eff['drop'] = eff['100%'] - eff['10%']
    return tlc, eff


def predict_split(model, split='test'):
    """True labels and predicted class probabilities for every crop of a split."""
    return DF[DF.split == split].label.to_numpy(), model.predict(make_ds(split), verbose=0)


def final_test(name):
    """Load the best checkpoint of the chosen model and score it ONCE on the held-out test set.
    Saves final_metrics.json and classification_report.json. Returns (model, metrics, report, y_true, y_pred, proba)."""
    model = load(name)
    y_true, proba = predict_split(model, 'test')
    y_pred = proba.argmax(1)
    report = classification_report(y_true, y_pred, labels=list(range(4)), target_names=TARGET,
                                   zero_division=0, output_dict=True)
    metrics = {'chosen_model': name,
               'test_accuracy': float(np.mean(y_true == y_pred)),
               'test_macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
               'test_balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
               'test_unsafe_recall': float(np.mean([np.mean(y_pred[y_true == i] == i) for i in UNSAFE])),
               'number_test_crops': int(len(y_true))}
    (RESULTS / 'final_metrics.json').write_text(json.dumps(metrics, indent=2))
    (RESULTS / 'classification_report.json').write_text(json.dumps(report, indent=2))
    return model, metrics, report, y_true, y_pred, proba


# -----------------------------------------------------
# 6. Plots
# -----------------------------------------------------
def save_figure(fig, name, dpi=170):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=dpi)
    plt.show()


def plot_dataset(df):
    """Class balance per split, crop size distribution, and two random training crops per class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    pd.crosstab(df.split, df.label).rename(columns=dict(enumerate(TARGET))).plot.bar(ax=axes[0], rot=0)
    axes[0].set(title='Annotated PPE region class balance', ylabel='Number of crops')
    sns.boxplot(data=df, x='split', y='area', ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Box area distribution (log scale)')
    save_figure(fig, 'dataset_exploration.png')

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for k in range(4):
        pool = df[(df.split == 'train') & (df.label == k)]
        for j, (_, r) in enumerate(pool.sample(n=min(2, len(pool)), random_state=SEED).iterrows()):
            axes[j, k].imshow(read_crop(r))
            axes[j, k].set_title(TARGET[k])
            axes[j, k].axis('off')
    save_figure(fig, 'dataset_samples.png')


GROUP_COLORS = dict(zip(['architecture', 'activation', 'optimization', 'regularization', 'imbalance', 'loss', 'epochs',
                         'grid search'], plt.cm.tab10.colors))


def plot_ablation_summary(abl, ref, lo, hi):
    """Change in validation macro-F1 per controlled experiment, with the baseline seed range shaded as noise."""
    plot = abl[abl.group != 'baseline'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axvspan(lo - ref, hi - ref, color='grey', alpha=.25, label='baseline seed range (noise)')
    ax.barh(plot.change, plot.delta, color=[GROUP_COLORS[g] for g in plot.group])
    ax.axvline(0, color='black', lw=.8)
    for g, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=g)
    ax.set_xlabel('Validation macro-F1 minus baseline mean')
    ax.set_title('One change at a time: effect on validation macro-F1')
    ax.legend(loc='lower right', fontsize=8)
    save_figure(fig, 'ablation_summary.png')


def plot_ablation_curves(abl, baseline='abl_baseline_s1'):
    """Validation accuracy per epoch, one panel per group, plus the baseline's train vs validation accuracy."""
    groups = [g for g in GROUP_COLORS if g in set(abl.group)]
    rows = int(np.ceil((len(groups) + 1) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4.3 * rows))
    for ax in axes.ravel():
        ax.axis('off')
    for ax, g in zip(axes.ravel(), groups):
        ax.axis('on')
        if baseline in HISTORIES:
            v = HISTORIES[baseline]['val_accuracy']
            ax.plot(range(1, len(v) + 1), v, 'k--', label='Baseline')
        for name in abl.index[abl.group == g]:
            if name in HISTORIES:
                v = HISTORIES[name]['val_accuracy']
                ax.plot(range(1, len(v) + 1), v, label=abl.change[name])
        ax.set(title=g.capitalize(), xlabel='Epoch', ylabel='Validation accuracy', ylim=(0.2, 0.9))
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    ax = axes.ravel()[len(groups)]
    if baseline in HISTORIES:
        ax.axis('on')
        h = HISTORIES[baseline]
        ep = range(1, len(h['accuracy']) + 1)
        ax.plot(ep, h['accuracy'], marker='o', label='train (with augmentation and dropout)')
        ax.plot(ep, h['val_accuracy'], marker='o', label='validation')
        ax.set(title='Baseline: train vs validation accuracy', xlabel='Epoch', ylabel='Accuracy')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    save_figure(fig, 'ablation_curves.png')


def plot_grid_search(grid):
    """Heatmap of validation macro-F1 for every (learning rate, batch size) pair.
    grid: DataFrame with columns lr, batch, val_macro_f1."""
    table = grid.pivot(index='lr', columns='batch', values='val_macro_f1').sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(table, annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'validation macro-F1'})
    ax.set(xlabel='batch size', ylabel='learning rate (Adam)', title='Grid search: learning rate x batch size')
    ax.set_yticklabels([f'{v:g}' for v in table.index], rotation=0)
    save_figure(fig, 'grid_search.png')
    return table


def plot_epochs(name, baseline='abl_baseline_s1'):
    """Train vs validation loss and accuracy of a run trained without early stopping, with the epoch
    of the lowest validation loss (where early stopping would have kept the weights)."""
    if name not in HISTORIES:
        print(f'No history for {name}.')
        return
    h = HISTORIES[name]
    ep = np.arange(1, len(h['loss']) + 1)
    best = int(np.nanargmin(h['val_loss'])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric, label in [(axes[0], 'loss', 'Loss'), (axes[1], 'accuracy', 'Accuracy')]:
        ax.plot(ep, h[metric], marker='o', label='train')
        ax.plot(ep, h['val_' + metric], marker='o', label='validation')
        if metric == 'accuracy' and baseline in HISTORIES:
            v = HISTORIES[baseline]['val_accuracy']
            ax.plot(range(1, len(v) + 1), v, 'k--', label='validation, baseline (early stopping)')
        ax.axvline(best, color='grey', ls=':', label=f'lowest validation loss (epoch {best})')
        ax.set(xlabel='Epoch', ylabel=label, title=f'{label}: {len(ep)} epochs without early stopping')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    save_figure(fig, 'epochs_experiment.png')


def plot_transfer_comparison(tlc, eff):
    """Convergence curves with 100% and 10% of the data, and the data-efficiency bars."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, pct in zip(axes[:2], (100, 10)):
        for name in tlc.index[tlc.data == f'{pct}%']:
            if name in HISTORIES:
                v = HISTORIES[name]['val_accuracy']
                ax.plot(range(1, len(v) + 1), v, marker='o', label=tlc.setup[name])
        ax.set(title=f'Convergence, {pct}% of the training images', xlabel='Epoch', ylabel='Validation accuracy', ylim=(0.2, 1))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    eff[['10%', '100%']].plot.bar(ax=axes[2], rot=15)
    axes[2].set(title='Data efficiency', ylabel='Validation macro-F1', xlabel='', ylim=(0, 1))
    save_figure(fig, 'transfer_comparison.png')


def plot_model_comparison(bench, names):
    """Validation macro-F1 and unsafe-class recall for the given models."""
    order = bench.loc[names].sort_values('val_macro_f1')
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ypos - 0.2, order['val_macro_f1'], height=0.4, label='Macro-F1')
    ax.barh(ypos + 0.2, order['unsafe_recall'], height=0.4, label='Unsafe-class recall')
    ax.set_yticks(ypos, order.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Validation score')
    ax.set_title('Model comparison on the validation set')
    ax.legend(loc='lower right')
    save_figure(fig, 'model_comparison.png')


def plot_validation_curves(names):
    """Validation accuracy per epoch for the given models (epoch budgets differ between stages)."""
    names = [n for n in names if n in HISTORIES]
    if not names:
        print('No training histories available; set RETRAIN = True to produce learning curves.')
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        ax.plot(HISTORIES[name]['val_accuracy'], label=name, alpha=.75)
    ax.legend(fontsize=7, ncol=2)
    ax.set(xlabel='Epoch', ylabel='Validation accuracy', title='Training comparisons (note: epoch budgets differ)')
    save_figure(fig, 'validation_curves.png', dpi=180)


def plot_loss_curves(names):
    """Training vs validation loss per model: a widening gap signals over-fitting."""
    names = [n for n in names if n in HISTORIES]
    if not names:
        print('No saved training histories (the saved run predates history logging). '
              'Set RETRAIN = True once to record them; later loaded runs will then show these curves.')
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, name in zip(axes.ravel(), names):
        ax.axis('on')
        h = HISTORIES[name]
        epochs = np.arange(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'], marker='o', label='train loss')
        ax.plot(epochs, h['val_loss'], marker='o', label='validation loss')
        ax.set(title=name, xlabel='Epoch', ylabel='Loss')
        ax.grid(alpha=.3)
    axes.ravel()[0].legend()
    fig.suptitle('Training vs validation loss (training loss includes dropout, augmentation and L2)')
    save_figure(fig, 'loss_curves.png')


def plot_confusion_matrix(y_true, y_pred, title, save_name='test_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET, yticklabels=TARGET)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=25)
    plt.title(title)
    if save_name:
        save_figure(fig, save_name, dpi=180)
    else:
        plt.show()


def error_gallery(y_true, y_pred, proba, n=12):
    """Up to n misclassified test crops with the model's confidence."""
    test_rows = DF[DF.split == 'test'].reset_index(drop=True)
    wrong = np.flatnonzero(y_true != y_pred)
    examples = wrong[:n] if len(wrong) else np.arange(min(8, len(test_rows)))
    fig, axs = plt.subplots(3, 4, figsize=(13, 10))
    axs = np.ravel(axs)
    for ax in axs:
        ax.axis('off')
    for ax, idx in zip(axs, examples):
        ax.imshow(read_crop(test_rows.iloc[idx]))
        ax.set_title(f'True: {TARGET[y_true[idx]]}\nPred: {TARGET[y_pred[idx]]} ({proba[idx].max():.2f})', fontsize=9)
    save_figure(fig, 'error_gallery.png')


def show_predictions(rows, y_true, y_pred, proba):
    """Grid of crops (DF rows) with the true class, the predicted class and the confidence; wrong ones in red."""
    n = len(rows)
    cols = 8
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(2.2 * cols, 2.6 * int(np.ceil(n / cols))))
    for ax in np.ravel(axs):
        ax.axis('off')
    for ax, (_, r), t, p, pr in zip(np.ravel(axs), rows.iterrows(), y_true, y_pred, proba):
        ax.imshow(read_crop(r))
        ax.set_title(f'True: {TARGET[t]}\nPred: {TARGET[p]} ({pr.max():.2f})', fontsize=8,
                     color='black' if t == p else 'red')
    fig.tight_layout()
    plt.show()
