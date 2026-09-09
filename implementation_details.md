# Implementation Details

Each experiment gives a group of agents a question, a persona for each agent, and a network describing who communicates with whom. The agents express opinions, exchange messages, and update their opinions. The analysis then asks whether simple mathematical rules can predict those updates.

### Concepts

| Term | Meaning in this codebase |
| --- | --- |
| Agent | One member of a simulated society, represented by a persona, an opinion, and an inbox. A language model generates its answers and messages. |
| Persona | Text describing the agent's views or expertise; supplied in its prompts. |
| Spin, `s` | An opinion encoded as `+1` or `−1`. In saved spin samples, `0` marks a failed sample, not a third answer. |
| Regime, `mode` | Either subjective statements (AGREE/DISAGREE) or objective questions (A/B). |
| Interaction matrix, `J` | The communication network. In generation, `J[i, j]` describes the connection from sender `i` to receiver `j`. |
| Replica / episode | One trial of a society answering one question on one graph. `Replica` is its in-memory representation; an episode is the saved record. |
| Trajectory | The sequence of opinions over an episode. Repeated trials can produce different trajectories. |
| Round / step | One message-and-opinion update. Step 0 records opinions before any messages are exchanged. |
| `k` spin samples | Repeated answers to the same opinion prompt, combined into one opinion by majority vote. |
| `Pi` / `pi` | A Python callable that takes a prompt string and returns a response string. |
| Coupling | A fitted coefficient describing how strongly neighboring opinions affect a predicted update. |

Here, `n` is the number of agents and `T` is the number of update rounds. The main runs use `n = 32`, `T = 8`, and `k = 5`, giving **nine recorded opinion states**: the initial state plus eight updates. These are experiment settings, not universal array sizes.

## 1. Repository layout and data flow

```
lib/                    experiment generation (Python package + consolidation notebooks)
  datagen/              core synchronous dynamics (main experiments)
  datagen_async/        schedule-driven fully asynchronous dynamics
  datagen_frontier/     mixed frontier-model dynamics (OpenRouter)
  features.py           shared embedding / PCA feature loaders
  clean_data.ipynb      raw per-episode JSON -> data/clean_runs.json
  clean_data_seq.ipynb  raw async JSON      -> data/clean_runs_seq.json
data/                   question banks, personas, raw episodes, consolidated runs
  lowrank/              additional graph-family episodes and construction README
res/                    analysis notebooks (figures and prediction tables)
  utils.py              shared analysis library (loading, features, fits, metrics)
online_appendix/        static browser for the message-level data
```

```text
Question banks + personas + interaction graphs
                     |
                     v
            Generate conversations             lib/datagen*/
                     |
                     v
            Raw episode JSON files             data/models/, async/, frontier/
                     |
                     v
            Consolidate main / async runs       lib/clean_data*.ipynb
                     |
                     v
            Fit models and make figures        res/
```

The pipeline has three stages:

1. **Generation.** The command-line entry points in `lib/datagen*/collect_*.py` build a bank of agent societies (a persona list, a question, and an interaction graph J), advance each society through the opinion dynamics by repeatedly calling a language model, and write one raw JSON file per episode under `data/models/`, `data/async/`, or `data/frontier/`, together with a `manifest.json` recording the full configuration.
2. **Consolidation.** The notebooks `lib/clean_data.ipynb` and `lib/clean_data_seq.ipynb` flatten the raw episode files into the two flat records files `data/clean_runs.json` (synchronous runs, 9,600 episodes) and `data/clean_runs_seq.json` (asynchronous runs, 480 episodes), attaching the ground-truth answer (objective questions) or political-lean label (subjective questions) to every episode.
3. **Analysis.** The numbered notebooks in `res/` use the shared library `res/utils.py` to produce descriptive figures and prediction tables. Most read the consolidated files; the frontier notebook reads `data/frontier/` directly, and the generalization notebook combines the main consolidated runs with episode files from `data/lowrank/`. These additional datasets do not need a consolidation step. Notebook `4_prediction.ipynb` additionally writes the fitted coupling constants to `res/couplings.json`, which the downstream notebooks `5_temperaturesweep.ipynb` and `6_distribution.ipynb` read back.

Generation writes the experiment data; analysis reads it. The generation code does not need fitted results from `res/`.

### A three-agent example

Consider a subjective statement and three agents, numbered 0, 1, and 2:

```text
J = [[ 0, +1, -1],
     [+1,  0,  0],
     [-1,  0,  0]]
```

Agents 0 and 1 receive each other's messages as sources they tend to agree with. Agents 0 and 2 receive each other's messages as sources they tend to disagree with. Agents 1 and 2 have no direct connection. The sign describes the relationship between agents, not whether a message says AGREE or DISAGREE.

One round works as follows:

1. Start from the current opinions, for example `[+1, -1, +1]`. At step 0 these were sampled with empty inboxes.
2. Ask each sender to write a message consistent with its current opinion. Agent 0's same message goes to both agents 1 and 2.
3. Build the new inboxes. Agent 0 gets agent 1's message in its “tends to agree with” section and agent 2's in its “tends to disagree with” section.
4. Ask each agent for its opinion again, using its new inbox. With `k = 5`, samples `[+1, +1, -1, +1, -1]` produce an aggregated opinion of `+1`.
5. Save both the five samples and the aggregated opinion, then repeat for the next round.

These values illustrate the mechanics; they are not results from a recorded experiment. The functions implementing this sequence are described in Section 2.4.

## 2. Experiment generation: `lib/`

### 2.1 The language-model abstraction

The dynamics code calls `response = pi(prompt)`. It does not need to know which service produced the response. The callable's type, `Pi`, is defined in [samplers.py](lib/datagen/samplers.py).

Functions in [utils.py](lib/datagen/utils.py) construct this callable for different backends:

- `make_openai_pi` wraps the OpenAI chat-completions API (or any OpenAI-compatible endpoint via `base_url`) with exponential-backoff retries; the sampling temperature (0.7 by default) and output-token cap are fixed at construction.
- `make_together_pi` reuses `make_openai_pi` against Together's OpenAI-compatible endpoint (used for the Llama, Qwen, and Gemma runs).
- `make_mock_pi` is a deterministic offline stand-in (hash-based A/B or AGREE/DISAGREE answers and a canned message) used for smoke-testing the full pipeline without API access; the collection CLIs expose it via `--mock`.

The frontier experiments add a fourth factory, `make_openrouter_pi` in [openrouter.py](lib/datagen_frontier/openrouter.py), again delegating to `make_openai_pi` with OpenRouter's base URL. It also maps the `--reasoning` flag onto OpenRouter's unified reasoning parameter; the collection CLI defaults to `none` to request direct judgments.

Because all backends reduce to the same `Pi` signature, the dynamics code is entirely backend-agnostic.

### 2.2 Prompts and spin sampling: `lib/datagen/samplers.py`

[samplers.py](lib/datagen/samplers.py) implements the two elementary LM operations of the model, in two regimes:

- `spin_sampler` elicits a binary opinion ("spin"). In the **subjective** regime the agent is prompted with its persona, its current inbox, and the statement, and must answer with exactly one word, AGREE (+1) or DISAGREE (−1). In the **objective** regime the persona describes an expertise, the statement is a two-option (A/B) problem, and the response A maps to +1, B to −1. The objective instruction requests a direct, single-character judgment without reasoning. The code comments describe this as a way to limit latency and cost and encourage snap judgments.
- `message_sampler` elicits the two-sentence message an agent broadcasts to its neighbors, conditioned on the same context plus the agent's current spin (an undecided spin of 0 is broken uniformly at random for the message's stance). The message states the agent's stance and its strongest reason.

Both prompts present the inbox split into two sections — messages from sources the agent *tends to agree with* and *tends to disagree with*. This is how the sign of the interaction matrix J enters the dynamics: the split is performed at the receiver from the sign of J (Section 2.3), so the samplers themselves never see J. Response parsing differs by regime:

- `_parse_spin_subjective` looks for DISAGREE first, then AGREE, within the response text. It is more permissive than the requested one-word format.
- `_parse_spin_objective` accepts A or B, including longer text containing only one of those standalone answer labels. It raises if both labels appear or neither appears.

The dynamics layer retries parse failures twice. A persistent failure becomes spin `0`; the synchronous spin phase also records other failed calls as `0`. If an agent has spin `0` when asked to write a message, its message stance is chosen randomly from `−1` and `+1`.

### 2.3 Interaction graphs: `lib/datagen/graph.py`

[graph.py](lib/datagen/graph.py) constructs an `n × n` matrix `J`:

| `J[i, j]` | What receiver `j` sees from sender `i` |
| --- | --- |
| `+1` | A message in the “sources you tend to agree with” section. |
| `−1` | A message in the “sources you tend to disagree with” section. |
| `0` | No message from this sender. |

The constructors produce symmetric graphs: `J[i, j] = J[j, i]`. Diagonal entries are zero, so agents do not send messages to themselves.

The generation module provides two constructors:

- `sample_J_num_edges_symmetric` draws a symmetric random graph with a prescribed number of non-zero entries: each upper-triangular pair receives U_ij ~ Uniform(−1, 1), the largest-|U| pairs are kept, and each kept edge takes the sign of its U (so positive "agree" and negative "disagree" bonds are equally likely).
- `make_lattice_J` builds two regular graph benchmarks: a square grid with up to four neighbors per agent and a triangular grid with up to six (the square grid plus a down-left diagonal), with a uniform bond sign. Boundaries have fewer neighbors; the grids do not wrap around.

The `num_edges` argument counts **non-zero matrix entries**, so each undirected connection counts twice. For example, 112 entries represent 56 undirected connections; with 32 agents, the average degree is `112 / 32 = 3.5`.

The additional low-rank matrices are stored in the episodes under `data/lowrank/`; their original generator is not included in this module. Each has a 10-node core with 34 internal edges and 22 outer nodes attached to one core hub, giving 56 undirected edges and no isolated nodes. The six signed matrices have algebraic rank 11 and spectral participation ratios of 5.92–6.88. See [the low-rank README](data/lowrank/README.md) for the construction description and measured frustration fractions.

### 2.4 Synchronous forward dynamics: `lib/datagen/dynamics.py`

[dynamics.py](lib/datagen/dynamics.py) is the heart of the generation code. The unit of simulation is a `Replica`: one (statement, J) trial holding the persona list, the current agree/disagree inbox of every agent, and the full spin and message history. `run_forward_dynamics` advances a list of replicas together through `num_steps` synchronous rounds:

**Initialization:** `_run_spin_phase(..., use_inbox=False)` samples all agents with empty inboxes to obtain `s(0)`.

**Each update round:**

| Order | Function | What it does |
| --- | --- | --- |
| 1 | `_run_message_phase` | Requests one message per sender with outgoing connections, using its previous opinion and inbox. Broadcasts that message to its recipients. |
| 2 | `_build_inboxes` | Groups incoming messages by the sign of `J` and shuffles their order using a seed derived from replica name, step, and receiver. Called within the message phase. |
| 3 | `_run_spin_phase` | Requests `k` opinion samples per agent using the new inbox, then saves the samples and their majority vote. |

`_aggregate_spin` takes the sign of the sum of the samples. If the sum is zero, it uses the first sample; an empty sample list produces `0`.

The calls within each phase run concurrently in a shared `ThreadPoolExecutor`. **Every message is collected before any new opinion is sampled.** This ordering is what “synchronous” means here; parallel API calls do not make the simulation asynchronous.

`_save_replica` writes one `<name>.json` file containing the experiment context and histories. See the field reference in Section 3 for their shapes and time indices.

An earlier version sampled multiple messages per sender. The comment above `_run_message_phase` records this change, and legacy runs remain in the data.

### 2.5 The main-experiment CLI: `lib/datagen/collect_energy.py`

[collect_energy.py](lib/datagen/collect_energy.py) turns the pieces above into the train/test trajectory dataset (`python -m lib.datagen.collect_energy`). Its responsibilities:

- **Question banks.** `load_questions` reads `data/subj/{train,test}.jsonl` or `data/obj/{train,test}.jsonl` depending on `--mode` (subjective statements vs. objective A/B problems; objective rows must carry `choices`).
- **Personas.** `load_canonical_personas` loads `data/<subj|obj>/personas.json` in a fixed canonical order, so agent i is always persona i across every episode — this alignment is what lets the analysis attach per-agent persona features later.
- **Graph bank.** `build_graph_bank` draws, from a single seeded RNG, the eight random training graphs `J0..J7`, reuses the first four at test time ("seen"), draws four fresh test-only graphs `Jf0..Jf3`, and appends the square and triangular lattices to both splits. Defaults: n = 32 agents, 112 non-zero entries per J (average degree 3.5).
- **Replica grid.** `build_replicas` instantiates one `Replica` per (question × graph × trajectory), 4 trajectories per pair by default, named `<qid>__<graph>__rep<r>`, and builds the manifest mapping each name to its qid, split, graph id, and whether the graph was seen in training.
- **Execution.** The backend is resolved from the model name (a small registry plus name-based inference), `manifest.json` is written with the complete configuration, and `run_forward_dynamics` produces one JSON per replica under `data/models/<model-slug>/<mode>_energy/` (8 steps, k = 5, seed 0 in the main run configuration).

The four models of the main experiments — GPT-4o-mini, Llama-3-8B-Instruct, Qwen, and Gemma — are each collected by one invocation per regime of this single CLI.

### 2.6 Asynchronous dynamics: `lib/datagen_async/`

In an asynchronous run, agents update individually according to a saved schedule. Different agents can therefore have different amounts of recent information.

The implementation separates schedule creation from execution:

1. **Create the schedule** with [make_schedule.py](lib/datagen_async/make_schedule.py), using `python -m lib.datagen_async.make_schedule`. It makes no API calls. Each timestep has 1,000 fine cells; an agent fires in each cell with probability `p_step / 1000`. With `p_step = 0.5`, an agent fires 0.5 times per timestep on average. Same-cell events are randomly ordered.
2. **Rebuild the experiment** with [collect_energy_seq.py](lib/datagen_async/collect_energy_seq.py), using `python -m lib.datagen_async.collect_energy_seq`. `build_replicas_from_schedule` uses the schedule metadata to reconstruct the question map and graph bank, then creates a `SeqReplica` for each scheduled episode.
3. **Replay the events** with [dynamics_seq.py](lib/datagen_async/dynamics_seq.py). After the initial opinion samples, `_advance_replica_chain` processes one episode's events in order.

At each event, the firing agent first posts a message based on its latest opinion, then resamples its opinion from the current message board. The board retains the latest message on each edge, including messages from agents that have not fired recently.

Events within one episode remain sequential. Different episodes run concurrently, and their LM calls use a shared thread pool. A failed episode chain is not saved, and chain failures cause the run to raise an error.

Schedules are stored at `data/async/schedules/<mode>_firing_schedule.json`. Each replica's schedule RNG is derived from SHA-256 of the schedule seed and replica name. Using the same graph seed and parameters as the synchronous experiment reproduces its random graph bank; asynchronous schedules exclude lattices.

`SeqReplica` extends `Replica` with the schedule and event histories. The module reuses `_aggregate_spin`, `_build_inboxes`, `_run_spin_phase`, `_spin_call_with_retry`, and the prompt samplers from the synchronous implementation. Saved records retain the common history fields and add event-level information.

### 2.7 Frontier mixed-model dynamics: `lib/datagen_frontier/`

The frontier experiment tests the fitted update rules on societies mixing two frontier model families. [collect_frontier.py](lib/datagen_frontier/collect_frontier.py) (`python -m lib.datagen_frontier.collect_frontier`) runs n = 64 agents — 32 on each of two models (by default GPT-5.6-sol and DeepSeek-V4-Flash via OpenRouter) — on **one** fixed random graph J0 (224 non-zero matrix entries, or 112 undirected connections, preserving average degree 3.5), for 8 steps with k = 1 spin sample, one episode per question, over the pooled train + test questions of both regimes. With the block assignment, the 32-persona bank is tiled over the 64 agents, so agents i and i + 32 share a persona across the two families. The round order and prompt protocol follow `lib.datagen.collect_energy`; the agent count, model assignment, graph selection, and sampling settings differ.

[dynamics_mixed.py](lib/datagen_frontier/dynamics_mixed.py) makes this possible with a minimal extension: `MixedReplica` adds a per-agent model list, and the message/spin phases are copies of the synchronous ones in which every LM call is routed to the `Pi` of the *acting agent's* model (`_pi_for` on a dict of per-model `Pi`s built by `make_openrouter_pis`). The saved JSON additionally records `agent_models`. Outputs go to `data/frontier/<modelA>__<modelB>/<mode>_energy/`.

### 2.8 Consolidation and shared features

- [clean_data.ipynb](lib/clean_data.ipynb) globs every episode JSON under `data/models/*/*/`, normalizes it into a flat table, parses the `<question>__<graph>__rep<r>` naming into columns, joins the ground-truth answers (from the objective banks' `answer` field) and political-lean labels (from `data/subj/lean.json`), asserts that every episode received exactly one of the two labels, and writes `data/clean_runs.json` (also pushed to the Hugging Face Hub as `physics-of-agents/agent-opinions`). [clean_data_seq.ipynb](lib/clean_data_seq.ipynb) does the same for `data/async/` into `data/clean_runs_seq.json`.
- [features.py](lib/features.py) is a small shared feature layer: a numpy-SVD `PCAReducer` (no sklearn dependency) and loaders for the persona and question embedding files, used when preparing the precomputed `embedding_pca10` features stored in the question banks.

## 3. Data: `data/`

### 3.1 Reading one saved episode

The synchronous episode JSON mirrors the `Replica` object. Arrays are stored as nested JSON lists.

| Field | Meaning / shape |
| --- | --- |
| `meta` | Run metadata, including the episode name and agent count. |
| `personas` | One persona string per agent; list order defines agent indices. |
| `statement`, `mode`, `choices` | The question, regime, and A/B options when applicable. |
| `J` | Interaction matrix, shape `(n, n)`. |
| `spins_history` | Aggregated opinions, shape `(T + 1, n)`. Index 0 is the initial state. |
| `spins_raw_history` | Individual opinion samples, shape `(T + 1, n, k)`. |
| `messages_history` | `T` dictionaries. Entry 0 contains the messages used to produce opinion state 1. JSON keys such as `"0,2"` mean sender 0 → receiver 2. |

For a main run, `spins_raw_history[2][7][3]` is agent 7's fourth sample at step 2. `spins_history[2][7]` is that agent's aggregated opinion at the same step. Indices are zero-based.

Keep the two opinion representations distinct: prediction uses aggregated spins, while `opinions(run)` averages the raw samples for descriptive analysis. The five samples `[+1, +1, -1, +1, -1]` yield an aggregated spin of `+1` but a mean opinion of `0.2`.

### 3.2 Input and output files

Inputs prepared before generation:

- `data/subj/` and `data/obj/` — the subjective and objective question banks. `train.jsonl` / `test.jsonl` carry, per question, the qid, text, split, the A/B choices and correct `answer` (objective), pilot-response statistics, and the precomputed 10-component PCA embedding `embedding_pca10`. `personas.json` is the canonical 32-persona bank per regime; `persona_embeddings.json` and `question_embeddings*.json` hold the raw text embeddings; `subj/lean.json` maps each subjective qid to a left/right/ambiguous lean.
- `data/async/schedules/` — the pre-sampled firing schedules (Section 2.6).

Stored datasets and outputs:

- `data/models/<model>/<mode>_energy/` — raw synchronous episodes, one JSON per replica plus `manifest.json` (4 models × 2 regimes).
- `data/async/gpt-4o-mini/<mode>_energy_seq/` — raw asynchronous episodes.
- `data/frontier/<modelA>__<modelB>/<mode>_energy/` — raw frontier mixed-model episodes.
- `data/lowrank/gpt-4o-mini/ood_lowrank/<mode>_rank_frustration/` — 240 additional episodes across both regimes, each with its own `J` and spin histories. Each regime has six graphs × ten test questions × two replicas. The manifests include entries for trajectories absent from the release; the generalization notebook loads existing episode files and uses the question banks to verify their test split. Objective coverage is ten of the twenty test questions. [README.md](data/lowrank/README.md) describes these graphs.
- `data/clean_runs.json`, `data/clean_runs_seq.json` — the consolidated records used by the analysis: one row per episode with model, regime, statement, J, spin histories, replica index, and the ground-truth / lean label.
- `data/J_matrices.json` (and `data/async/J_matrices.json`) — the communication networks in exportable form.

## 4. Analysis: `res/`

The shared analysis functions live in [res/utils.py](res/utils.py). This module defines the main model identifiers, `N_AGENTS = 32`, the group-classification threshold `TAU = 0.2`, and the class names used in plots.

### 4.1 Load trajectories and build features

`load_data` reads consolidated runs, question banks, and persona embeddings. Embeddings are numeric representations of text; PCA reduces them to a few coordinates used as prediction features. Questions use their first three stored PCA components, and `pca3` computes three persona components per regime.

`graph_classes` labels graphs as seen, train-only, fresh, or lattice from their structure and the question splits in which they occur. `opinions` returns the mean of the raw spin samples, with shape `(T + 1, n)` for one episode.

The generalization notebook labels its additional low-rank graphs from their source directory and identifies square and triangular lattices by exact matrix comparison. This keeps all-positive low-rank graphs distinct from lattices. It checks that training and evaluation share neither question IDs nor graph matrices, and displays the question, graph, replica, and episode counts for every family.

`static_field` constructs features that stay fixed throughout an episode:

| Feature block | Number of columns | Meaning |
| --- | --- | --- |
| Bias | 1 | A constant feature for the baseline tendency toward `+1`. |
| Persona, `p` | 3 | The agent's persona coordinates. |
| Question, `q` | 3 | The question's coordinates, repeated for each agent. |
| Persona–question products, `p ⊗ q` | 9 | Every persona coordinate multiplied by every question coordinate. |
| **Total** | **16** | The static feature block. |

`build_episodes` groups one model × regime into aligned arrays. `build_xy` then pairs each previous opinion with its next opinion:

- `x` has shape `(episodes, T, n, 19)`: one previous spin, 16 static features, and two peer drives.
- `y` has shape `(episodes, T, n)`: the next aggregated spin. Unparsed targets (`0`) are excluded from fitting and scoring.

A **peer drive** sums neighboring opinions with the corresponding connection weights. The positive drive is `J⁺s`, with `J⁺ = max(J, 0)`. The negative drive is `J⁻s`, with `J⁻ = min(J, 0)`, so its weights retain their negative signs. The analysis uses matrix–vector products; the generated graphs are symmetric, so this agrees with the sender/receiver convention used during generation.

### 4.2 Fit an update rule and run it forward

`Logistic` fits a logistic regression: it maps the features to a probability of the next spin being `+1`. Its implementation standardizes feature columns and uses full-batch `adam` to optimize cross-entropy with a small L2 penalty (`10⁻³`) on the persona/question field weights. The bias and coupling coefficients are unpenalized. Couplings start at `0.1`.

The logistic loss is convex, but this alone does not guarantee a unique solution or convergence of the finite Adam run.

`fit_continuous` in the synchronous and asynchronous prediction notebooks fits the original feature columns without centering or scaling. Its L2 penalty (`10⁻³`) applies to the non-bias field coefficients in these original coordinates. Couplings start at `0.1` and are unpenalized. All other optimization parameters start at zero, including `θ₀`, so the initial update rate is `ε = sigmoid(0) = 0.5`. The sigmoid keeps the fitted update rate in `(0, 1)`.

The continuous fitter returns its coefficients directly; one-step predictions and rollouts use the original features. The discrete logistic fitter converts its standardized coefficients and intercept back to the original coordinates before returning them. For a two-stage continuous fit, the first stage's unsigned coefficient supplies a fixed additive score in the second stage. The continuous objective jointly fits `ε` and the response weights and does not inherit the logistic model's convexity guarantee.

`two_stage_fit` handles designs whose peer drives are linearly dependent. In particular, the unsigned drive satisfies `|J|s = J⁺s − J⁻s`. It first fits the unsigned coupling `β₀`, then keeps that contribution fixed while fitting the signed couplings.

There are two ways to evaluate an update rule:

| Evaluation | Where the previous opinions come from |
| --- | --- |
| One-step prediction | The recorded experiment, at each step. |
| Rollout | The recorded initial state, followed by the model's own predictions. Errors can accumulate across steps. |

`drives` computes peer-drive features, and `discrete_rollout` applies the fitted rule repeatedly. A deterministic rollout chooses a spin from the sign of the fitted score; a stochastic rollout samples a spin using the predicted probability.

### 4.3 Score predictions and describe behavior

`raw_acc` is the percentage of correct predictions among transitions with parsed targets.

`fcba` is **flip-and-class balanced accuracy**. It computes accuracy separately for four groups, then averages the nonempty groups equally:

| Transition | Next opinion |
| --- | --- |
| Flip | `+1` |
| Flip | `−1` |
| Stay | `+1` |
| Stay | `−1` |

This keeps frequent “stay” events from dominating the score. Both previous and next spins must be nonzero. When all four groups are present, always retaining the previous spin or always predicting one label scores 50%.

For descriptive plots, `flip_counts` and `individual_archetypes` classify each agent by sign switches: Frozen (0), Switcher (1), Intermittent (2), or Oscillating (3 or more). Zero-valued opinions carry the last sign when counting switches.

`group_archetypes` compares the initial and final mean opinion against a split band around zero. It labels the society Persistent Split, Convergence, Divergence, Majority Switch, or Persistent Majority. The threshold is `TAU = 0.2`; the default uses `|mean opinion| < TAU`, while `closed=True` includes the boundary.

### 4.4 Find the notebook for a question

**Execution dependency:** `4_prediction.ipynb` writes `res/couplings.json`. Run that notebook before `5_temperaturesweep.ipynb` or `6_distribution.ipynb` if the fitted file is absent or needs regenerating. `4_prediction_generalization.ipynb` fits its own models and can run independently from the repository root or `res/`; it does not read or overwrite `couplings.json`.

The notebooks, in reading order:

- [1_archetypes.ipynb](res/1_archetypes.ipynb) — the individual and group trajectory archetypes and their composition tables across models and regimes.
- [2_conviction.ipynb](res/2_conviction.ipynb) — the mean-conviction plane (net opinion vs. conviction) over rounds and the consensus / polarization / indifference decomposition.
- [3_truth_seeking.ipynb](res/3_truth_seeking.ipynb) — objective questions: does the net opinion's sign converge to the correct answer over the 9 rounds. [3S_politicallean.ipynb](res/3S_politicallean.ipynb) and [3S_labelbias.ipynb](res/3S_labelbias.ipynb) analyze political lean and answer-label bias. The objective bank has balanced answer labels.
- [4_prediction.ipynb](res/4_prediction.ipynb) — the main prediction table: the update rules (persistence, interaction-free, mean-field Curie–Weiss, the discrete update, and its three-coupling extension) fit on train questions × the eight random graphs and scored on test questions × seen vs. fresh graphs, under both metrics, one-step and rollout. It ends by refitting the discrete update at one / three / five couplings and **writing `res/couplings.json`**, the interface to the two notebooks below.
- [4_prediction_generalization.ipynb](res/4_prediction_generalization.ipynb) — fits the five discrete methods once per GPT-4o-mini question regime on training questions × eight random graphs, then evaluates held-out questions on four fresh random graphs, six low-rank graphs, and the square and triangular lattices. It reads low-rank episodes directly from `data/lowrank/` and reports coverage before scoring. Its 80 scores (five methods × two regimes × four graph families × one-step/rollout) use flip-and-class balanced accuracy and are displayed as a table and exported as LaTeX.
- [5_temperaturesweep.ipynb](res/5_temperaturesweep.ipynb) — reads `couplings.json`, varies the temperature of the fitted three-coupling update to study changes in collective behavior. Here temperature controls the randomness of the fitted model, rather than the API sampling temperature. The notebook locates the susceptibility peak separately for each question–graph group, then reports the mean of those group peak temperatures. It also plots the mean susceptibility curve and its standard deviation across groups.
- [6_distribution.ipynb](res/6_distribution.ipynb) — reads `couplings.json` and asks whether free-running the fitted update from the real s(0) reproduces the *distribution* of individual and group archetypes of the real runs, plus the replica predictability ceiling.
- [7_prediction_continuous.ipynb](res/7_prediction_continuous.ipynb) — the continuous-time extension `m(t+1) = (1−ε)s(t) + ε·tanh(w·design)`, which blends the previous spin with a fitted response. It is fit by penalized maximum likelihood on the original feature columns with the shared Adam optimizer and added to the prediction tables.
- [8_prediction_async.ipynb](res/8_prediction_async.ipynb) — the continuous-time methods refit on the schedule-driven asynchronous runs (`data/clean_runs_seq.json`).
- [4_prediction_frontier.ipynb](res/4_prediction_frontier.ipynb) — the prediction methods on the frontier mixed-family run (read directly from `data/frontier/`): n = 64, one graph, personas tiled, k = 1; with a single J there is no seen/fresh axis, so the table's columns report model families instead.
