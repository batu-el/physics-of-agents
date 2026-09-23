# Physics of Agents

Code for [Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents](https://arxiv.org/abs/2608.16578).

Each experiment gives a group of agents a question, a persona for each agent, and a network describing who communicates with whom. The agents express opinions, exchange messages, and update their opinions. The analysis then asks whether simple mathematical rules can predict those updates.

```
lib/                    experiment / generation
  datagen/              synchronous dynamics (main experiments)
  datagen_async/        schedule-driven asynchronous dynamics
  datagen_frontier/     mixed frontier-model dynamics
  features.py           shared embedding / PCA feature loaders
  clean_data.ipynb      raw per-episode JSON -> data/clean_runs.json
  clean_data_seq.ipynb  raw async JSON      -> data/clean_runs_seq.json
data/                   question banks, personas, raw episodes, consolidated runs
res/                    analysis notebooks (figures and prediction tables)
  utils.py              shared analysis library (loading, features, fits, metrics)
online_appendix/        static browser for the message-level data
```

See [implementation_details.md](implementation_details.md) for the code and data guide, and [reproducibility_audit.md](reproducibility_audit.md) for the reproducibility audit.

### System Requirements

This code uses Python and the packages listed in [requirements.txt](requirements.txt). The analysis notebooks under res/ uses CPU computation and do not require a GPU or other specialized hardware. Generating new language-model responses with the data generation code under lib/ requires access to a language model API. The code has been successfully tested on macOS 26.6.2, Apple Silicon (arm64), Python 3.9.13. 

### Installation Guide
```bash
git clone https://github.com/batu-el/physics-of-agents.git
cd physics-of-agents

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installation takes approximately 5 minutes on a MacBook Pro with an Apple M1 Pro chip and 16 GB RAM. This includes approximately 30 seconds to create a fresh Python environment and download and install dependencies, plus 4 minutes 30 seconds to clone the repository.

### Demo
Instructions for running the analysis: The data collected in our study is available under data/. Run the notebooks under res/ to follow the analysis step by step.

Instructions for running data generation: For generating the data, use the files under lib/. The following command will use the first question from the training and test sets for the subjective dataset (`data/subj/train.jsonl`, `data/subj/test.jsonl`). It uses the first four personas from `data/subj/personas.json`, one shared graph, two update rounds, and three opinion samples per
agent. `--num-edges 6` specifies six nonzero entries in the symmetric interaction matrix, corresponding to three undirected connections. This demo uses simulated responses through the --mock option, requires no API key, and takes approximately one second on the tested machine. To generate responses using gpt-4o-mini, remove --mock and set the OPENAI_API_KEY environment variable. Runtime for API-backed generation depends on response times and rate limits.

```bash
python -m lib.datagen.collect_energy \
  --mock \
  --mode subjective \
  --limit 1 \
  --num-agents 4 \
  --num-edges 6 \
  --num-train-graphs 1 \
  --num-seen-graphs 1 \
  --num-fresh-graphs 0 \
  --no-lattices \
  --trajectories 1 \
  --num-steps 2 \
  --k 3 \
  --max-workers 1 \
  --seed 0 \
  --output-dir demo_output
```

For API-backed generation only

```bash
export OPENAI_API_KEY="your-api-key"
```

Expected output: The command creates 3 files:
```text
demo_output/
  manifest.json
  political_stance_0__J0__rep00.json
  political_stance_156__J0__rep00.json
```
The manifest describes two episodes: one training question and one test question. Each episode contains a 4 × 4 interaction matrix `J`, a 3 × 4 array containing the initial opinions and the opinions after each of the two updates in `spins_history`, a 3 × 4 × 3 array of individual opinion samples in `spins_raw_history`, two rounds of messages `messages_history`. 

### Instructions for use
Open a notebook in `res/` and run its cells in order. A useful starting point is `res/1_archetypes.ipynb`, which loads the released synchronous data and produces trajectory-classification summaries. Its initial data-loading cell should report `9600 societies`.

For prediction and downstream analyses:
- Run `res/4_prediction.ipynb` to fit the prediction models and regenerate `res/couplings.json`. 
- Run `res/5_temperaturesweep.ipynb` and `res/6_distribution.ipynb` using those fitted couplings.

### Running Custom Experiments
For running new experiments, prepare separate training and test JSONL files. Each line must contain a unique `qid` and a `question`.

Example `my_train.jsonl`:

```json
{"qid":"custom_train_0","question":"Public transport should be free."}
```

Example `my_test.jsonl`:

```json
{"qid":"custom_test_0","question":"Cities should create more pedestrian-only streets."}
```

Prepare a persona file, `my_personas.json`, containing a JSON list:

```json
[
  "You prioritize environmental protection.",
  "You prioritize individual freedom.",
  "You prioritize economic efficiency.",
  "You prioritize social equality."
]
```

Run the offline demo command above with these additional arguments and replace its output directory with `custom_output`.
```text
--train-file my_train.jsonl
--test-file my_test.jsonl
--personas my_personas.json
--output-dir custom_output
```

Keep question IDs distinct across both files, since they are used in episode filenames.

For objective experiments, use `--mode objective` and include a `choices` object with `A` and `B` alternatives.

```json
{"qid":"custom_objective_0","question":"What is 2 + 2?","choices":{"A":"4","B":"5"},"answer":"A"}
```

### License

The source code is distributed under the
[Apache License 2.0](LICENSE).