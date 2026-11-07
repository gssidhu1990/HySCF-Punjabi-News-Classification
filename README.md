# HySCF-Punjabi-News-Classification
Punjabi-News-Classification Repository Contains trained model and Code used in this study


# Punjabi News Classification

This project classifies Punjabi news into **10 categories**. The demo uses the trained **HySCF F06 model from seed 42**. It runs on CPU without retraining.

## Repository files

| File or folder | Contents |
| --- | --- |
| `code/` | Python code and the CPU prediction example. |
| `dataset/` | Dataset files and article metadata. |
| `experiments/` | Experiment records and results. |
| `model Trained 42 seed/` | Saved model files for the seed 42 demo. |
| `LICENSE` | Repository license. |
| `README.md` | Instructions for using the project. |

## Run on Kaggle

1. Open a Python notebook using **CPU** and attach the saved `mymodel` dataset.
2. Open the CPU prediction code in `code/`. Paste the complete code into one notebook cell. The code uses TensorFlow/Keras, NumPy and scikit-learn. Use the package versions saved with the model, if provided.
3. Check the `ROOT` path near the top of the code. It should point to the folder that contains both `experiments/` and `preprocessors/`. The supplied path is:

   ```python
   ROOT = Path(
       "/kaggle/input/datasets/drgurpreetsidhu/mymodel/"
       "HySCF_SixFusion_5Seed_Components"
   )
   ```

   If your model is attached at a different location, update `ROOT`. The code looks for these four files under that folder:

   - `experiments/F06_PARALLEL_FLAT_CNN__seed42/model.keras`
   - `preprocessors/tokenizer.json`
   - `preprocessors/tfidf_5000/vectorizer.pkl`
   - `preprocessors/label_mapping.json`

4. At the bottom of the cell, enter your Punjabi headline in `TITLE` and the article body in `CONTENT`. Run the whole cell. You can also run it as supplied to try the example article.

The program prints the predicted category, its confidence score and the five highest category scores. The training dataset is not required to classify a new article.

**If a file is marked `MISSING`:** Check `ROOT` and the four paths above. Keep the model, tokenizer, vectorizer and label mapping from the same saved run.

See `LICENSE` for the repository’s license terms.
