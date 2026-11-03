import random
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42

POSSIBLE_ZIPS = [
    Path("/mnt/data/Datasets (2).zip"),
    Path("/content/Datasets.zip"),
    Path("Datasets.zip"),
]
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_TO_INT = {"benign": 0, "malignant": 1}
OUTPUT_DIR = Path("model_outputs")

HOG_IMG_SIZE = (128, 128)

IMG_SHAPE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

BASELINE_NAME = "Baseline: HOG + SVM"
IMPROVED_NAME = "Improved: MobileNetV2 fine-tuned"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def locate_zip() -> Path:
    zip_path = next((p for p in POSSIBLE_ZIPS if p.exists()), None)
    if zip_path is None:
        try:
            from google.colab import files

            uploaded = files.upload()
            zip_path = Path(next(iter(uploaded.keys())))
        except Exception as e:
            raise FileNotFoundError(
                "Put Datasets.zip in the runtime or upload it in Colab."
            ) from e
    return zip_path


def extract_dataset(zip_path: Path) -> Path:
    extract_dir = (
        Path("/content/dataset_extracted")
        if Path("/content").exists()
        else Path("./dataset_extracted")
    )
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


    data_dir = None
    for p in [extract_dir, *extract_dir.rglob("*")]:
        if p.is_dir() and (p / "benign").is_dir() and (p / "malignant").is_dir():
            data_dir = p
            break

    if data_dir is None:
        raise FileNotFoundError(
            "Could not find folders named benign and malignant after extraction."
        )

    print("Dataset folder:", data_dir)
    return data_dir


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    records = []
    for label in ["benign", "malignant"]:
        folder = data_dir / label
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                records.append({"path": str(p), "label": label})

    df = pd.DataFrame(records).drop_duplicates("path").reset_index(drop=True)
    if df.empty:
        raise ValueError("No supported image files found in benign/malignant folders.")

    print(df["label"].value_counts())
    print(df.head())
    return df


def show_samples(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    samples = df.sample(min(8, len(df)), random_state=SEED)
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        im = Image.open(row["path"]).convert("RGB")
        ax.imshow(im)
        ax.set_title(row["label"])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def distribution_error(frame: pd.DataFrame, idx, overall: dict) -> float:
    d = frame.iloc[idx]["label"].value_counts(normalize=True).to_dict()
    return sum(abs(d.get(k, 0) - v) for k, v in overall.items())


def split_dataset(df: pd.DataFrame):
    df["group_id"] = df["path"].map(lambda s: Path(s).stem.split("_")[0])


    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(df["path"], df["label"], groups=df["group_id"]))


    overall = df["label"].value_counts(normalize=True).to_dict()
    test_fold = min(
        range(len(splits)), key=lambda i: distribution_error(df, splits[i][1], overall)
    )
    trainval_idx, test_idx = splits[test_fold]
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)


    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    inner_splits = list(
        sgkf2.split(trainval_df["path"], trainval_df["label"], groups=trainval_df["group_id"])
    )
    val_fold = min(
        range(len(inner_splits)),
        key=lambda i: distribution_error(trainval_df, inner_splits[i][1], overall),
    )
    train_idx, val_idx = inner_splits[val_fold]
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name}: n={len(part)}")
        print(part["label"].value_counts())
    print("\nGroup overlap checks:")
    print("train/test overlap:", len(set(train_df.group_id) & set(test_df.group_id)))
    print("validation/test overlap:", len(set(val_df.group_id) & set(test_df.group_id)))
    assert not (set(train_df.group_id) & set(test_df.group_id))
    assert not (set(val_df.group_id) & set(test_df.group_id))


    OUTPUT_DIR.mkdir(exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    return train_df, val_df, test_df


def hog_features(paths) -> np.ndarray:
    feats = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image)
        arr = resize(arr, (*HOG_IMG_SIZE, 3), anti_aliasing=True, preserve_range=True).astype(
            np.uint8
        )
        gray = rgb2gray(arr)
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        feats.append(feat)
    return np.asarray(feats, dtype=np.float32)


def train_baseline(X_train_hog, y_train, X_val_hog, y_val) -> Pipeline:
    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=3,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    baseline.fit(X_train_hog, y_train)

    baseline_val_prob = baseline.predict_proba(X_val_hog)[:, 1]
    baseline_val_pred = (baseline_val_prob >= 0.5).astype(int)
    print("Baseline validation report (not test):")
    print(
        classification_report(
            y_val, baseline_val_pred, target_names=["benign", "malignant"], zero_division=0
        )
    )
    return baseline


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SHAPE)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(frame: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].map(LABEL_TO_INT).astype(np.int32).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        aug = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.10),
                layers.RandomContrast(0.10),
            ]
        )
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def build_improved_model():
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SHAPE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SHAPE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    improved = keras.Model(inputs, outputs)
    return improved, base


def train_head(improved, train_ds, val_ds, class_weight, callbacks):
    compile_model(improved, learning_rate=1e-3)
    return improved.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def fine_tune(improved, base, train_ds, val_ds, class_weight, callbacks):
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(improved, learning_rate=1e-5)
    return improved.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def plot_learning_curves(history1, history2) -> None:
    hist = {}
    for h in [history1, history2]:
        for k, vals in h.history.items():
            hist.setdefault(k, []).extend(vals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.get("loss", []), label="train loss")
    axes[0].plot(hist.get("val_loss", []), label="validation loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(hist.get("pr_auc", []), label="train PR-AUC")
    axes[1].plot(hist.get("val_pr_auc", []), label="validation PR-AUC")
    axes[1].set_title("PR-AUC")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def metric_row(y_true, prob, model_name, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, pred),
        "Precision (malignant)": precision_score(y_true, pred, zero_division=0),
        "Recall / Sensitivity (malignant)": recall_score(y_true, pred, zero_division=0),
        "Specificity (benign)": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1 (malignant)": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "PR-AUC": average_precision_score(y_true, prob),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def evaluate_models(y_test, baseline_test_prob, improved_test_prob) -> pd.DataFrame:
    results = pd.DataFrame(
        [
            metric_row(y_test, baseline_test_prob, BASELINE_NAME),
            metric_row(y_test, improved_test_prob, IMPROVED_NAME),
        ]
    )
    print(results.round(4))

    for name, prob in [
        (BASELINE_NAME, baseline_test_prob),
        (IMPROVED_NAME, improved_test_prob),
    ]:
        pred = (prob >= 0.5).astype(int)
        print("\n", name)
        print(
            classification_report(
                y_test,
                pred,
                labels=[0, 1],
                target_names=["benign", "malignant"],
                zero_division=0,
            )
        )
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm)
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1], ["benign", "malignant"])
        ax.set_yticks([0, 1], ["benign", "malignant"])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        plt.tight_layout()
        plt.show()

    return results


def bootstrap_ci(y_true, prob, metric_fn, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            values.append(metric_fn(y_true[idx], prob[idx]))
        except Exception:
            continue
    return np.quantile(values, [0.025, 0.975])


def compute_confidence_intervals(y_test, baseline_test_prob, improved_test_prob) -> pd.DataFrame:
    metric_fns = {
        "Accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
        "F1 (malignant)": lambda y, p: f1_score(y, p >= 0.5, zero_division=0),
        "ROC-AUC": lambda y, p: roc_auc_score(y, p),
        "PR-AUC": lambda y, p: average_precision_score(y, p),
    }
    ci_rows = []
    for model_name, probs in [
        (BASELINE_NAME, baseline_test_prob),
        (IMPROVED_NAME, improved_test_prob),
    ]:
        for metric_name, fn in metric_fns.items():
            lo, hi = bootstrap_ci(y_test, probs, fn)
            ci_rows.append(
                {
                    "Model": model_name,
                    "Metric": metric_name,
                    "95% CI lower": lo,
                    "95% CI upper": hi,
                }
            )
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.round(4))
    return ci_df


def paired_comparison(y_test, baseline_test_prob, improved_test_prob) -> None:
    base_pred = (baseline_test_prob >= 0.5).astype(int)
    deep_pred = (improved_test_prob >= 0.5).astype(int)

    base_correct = base_pred == y_test
    deep_correct = deep_pred == y_test


    b = int(np.sum((~base_correct) & deep_correct))
    c = int(np.sum(base_correct & (~deep_correct)))
    table = [
        [int(np.sum(base_correct & deep_correct)), c],
        [b, int(np.sum((~base_correct) & (~deep_correct)))],
    ]
    test_result = mcnemar(table, exact=False, correction=True)

    print("Paired correctness table [[both correct, baseline only correct],")
    print("                         [deep only correct, both wrong]]:")
    print(table)
    print(f"McNemar statistic = {test_result.statistic:.4f}")
    print(f"McNemar p-value = {test_result.pvalue:.6f}")
    print(
        f"Accuracy difference (improved - baseline) = "
        f"{accuracy_score(y_test, deep_pred) - accuracy_score(y_test, base_pred):.4f}"
    )

    if test_result.pvalue < 0.05:
        print("At alpha=0.05, error patterns differ significantly between models.")
    else:
        print("At alpha=0.05, there is not enough evidence that error patterns differ.")
    print("This does not prove clinical usefulness; examine malignant recall, false negatives,")
    print("confidence intervals, and external validation.")


def save_outputs(
    baseline,
    improved,
    test_df,
    y_test,
    baseline_test_prob,
    improved_test_prob,
    results,
    ci_df,
) -> None:
    joblib.dump(baseline, OUTPUT_DIR / "baseline_hog_svm.joblib")
    improved.save(OUTPUT_DIR / "improved_mobilenetv2.keras")

    predictions = test_df[["path", "label"]].copy()
    predictions["y_true"] = y_test
    predictions["baseline_probability_malignant"] = baseline_test_prob
    predictions["improved_probability_malignant"] = improved_test_prob
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    results.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    ci_df.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False)

    print("Saved artifacts in:", OUTPUT_DIR.resolve())
    print("Files:", sorted(p.name for p in OUTPUT_DIR.iterdir()))


def main() -> None:
    set_seed()
    data_dir = extract_dataset(locate_zip())


    df = build_dataframe(data_dir)
    show_samples(df)


    train_df, val_df, test_df = split_dataset(df)


    X_train_hog = hog_features(train_df["path"])
    X_val_hog = hog_features(val_df["path"])
    X_test_hog = hog_features(test_df["path"])
    y_train = train_df["label"].map(LABEL_TO_INT).to_numpy()
    y_val = val_df["label"].map(LABEL_TO_INT).to_numpy()
    y_test = test_df["label"].map(LABEL_TO_INT).to_numpy()

    baseline = train_baseline(X_train_hog, y_train, X_val_hog, y_val)


    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)


    counts = train_df["label"].map(LABEL_TO_INT).value_counts().to_dict()
    total = sum(counts.values())
    class_weight = {cls: total / (2 * counts.get(cls, 1)) for cls in [0, 1]}
    print("Class weights:", class_weight)

    improved, base = build_improved_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "mobilenet_best.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history1 = train_head(improved, train_ds, val_ds, class_weight, callbacks)


    history2 = fine_tune(improved, base, train_ds, val_ds, class_weight, callbacks)
    plot_learning_curves(history1, history2)


    baseline_test_prob = baseline.predict_proba(X_test_hog)[:, 1]

    improved_test_prob = improved.predict(test_ds, verbose=1).ravel()
    results = evaluate_models(y_test, baseline_test_prob, improved_test_prob)


    ci_df = compute_confidence_intervals(y_test, baseline_test_prob, improved_test_prob)


    paired_comparison(y_test, baseline_test_prob, improved_test_prob)


    save_outputs(
        baseline,
        improved,
        test_df,
        y_test,
        baseline_test_prob,
        improved_test_prob,
        results,
        ci_df,
    )


if __name__ == "__main__":
    main()
