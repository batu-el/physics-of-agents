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
