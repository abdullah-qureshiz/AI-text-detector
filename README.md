# AI Text Detector

A beginner-friendly Python command-line project that estimates whether a piece of
writing looks **AI-generated** or **human-written**.

The program checks each text with two detectors:

1. **Rule-based detector** - looks at writing statistics such as sentence length
   and vocabulary diversity.
2. **Machine-learning detector** - uses a trained scikit-learn model with
   TF-IDF features and logistic regression.

> **Important:** These results are estimates, not proof. Text detectors can be
> wrong, especially with short text or writing from non-native English speakers.

## Requirements

- Python 3.9 or newer
- pip

The required Python packages are listed in [`requirements.txt`](requirements.txt).

## Project files

| File or folder | Purpose |
| --- | --- |
| [`main.py`](main.py) | Starts the interactive command-line application |
| [`detectors.py`](detectors.py) | Contains the shared detector code and both detector classes |
| [`config.py`](config.py) | Loads configuration values |
| [`train.py`](train.py) | Trains and saves the machine-learning model |
| [`make_sample_data.py`](make_sample_data.py) | Creates a small demo dataset |
| [`data/sample_data.csv`](data/sample_data.csv) | Sample training data |
| [`models/text_model.joblib`](models/text_model.joblib) | Saved machine-learning model |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Quick start on Windows

Open PowerShell in the project folder:

```powershell
cd "D:\ai eng BOOTCAMP\python\Project 1st"
```

### 1. Create a virtual environment

If you do not already have one:

```powershell
py -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the program with your installed
Python instead, or allow local scripts with:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the detector

```powershell
python main.py
```

Paste or type your text. Type `END` on its own line when you are finished.
The program needs at least **30 words** and accepts up to **20,000 characters**.

Example:

```text
Paste your text. Type END on a new line when finished:
This is a sample paragraph with enough words for the detector to analyze ...
END
```

After both detectors show their results, enter `y` to check another text or
enter `n` to exit.

## Train the model yourself

The repository already includes sample data and a saved model, so training is
optional. To recreate the demo dataset and train a new model:

```powershell
python make_sample_data.py
python train.py
```

The training script expects a CSV file with exactly these columns:

```text
text,label
```

Labels use `0` for human-written text and `1` for AI-generated text. The
trained model is saved to `models/text_model.joblib`.

The included dataset is intentionally small and made for demonstration only.
For meaningful results, use a large, diverse, high-quality dataset.

## Configuration

The default settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DATA_PATH` | `data/sample_data.csv` | Training CSV location |
| `MODEL_PATH` | `models/text_model.joblib` | Saved model location |
| `MIN_WORDS` | `30` | Minimum words required for analysis |
| `MAX_CHARS` | `20000` | Maximum input length |
| `THRESHOLD` | `0.5` | Probability threshold for the AI label |

To override a setting, create a local `.env` file in the project folder. For
example:

```env
MIN_WORDS=20
MAX_CHARS=20000
THRESHOLD=0.5
```

## How it works

1. The input is checked for length and minimum word count.
2. Extra whitespace is cleaned up.
3. The rule-based detector calculates simple writing statistics.
4. The machine-learning detector loads the saved model and calculates an AI
   probability.
5. Each detector displays a label, confidence percentage, and details.

## Limitations

- The detector does not prove who wrote a text.
- Short, edited, translated, or unusual writing may be classified incorrectly.
- The demo model is trained on synthetic sample sentences and should not be
  used for high-stakes decisions.
- The machine-learning detector requires the model file. If it is missing, run
  `python train.py`.

## Learning ideas for beginners

- Change the minimum word limit in `.env`.
- Add more examples to the dataset.
- Compare the two detector results.
- Add a third detector by creating another class that follows the
  `BaseTextDetector` interface.
