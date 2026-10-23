# Running MACS on ISCA (University of Exeter GPUs)

Why: the decision rule pre-registered on 15 October 2026 says that if a University GPU is
available by 26 October, the main run uses Qwen3.5-27B. Access arrived on 22 October, so the
27B model becomes the main model; 9B is run again under the same design as a second full-design
model. Results from different models are never pooled.

Account for every job: `COMM514`. The 27B model needs a full A100 (`-p gpu`); a 10 GB MIG
slice (`gpu-mig`) is too small for it.

## 1. Log in (once per session)

1. Off campus, connect the University VPN first.
2. On the laptop, open PowerShell and run (replace `uid123` with your University username):

       ssh uid123@login.isca.ex.ac.uk

   The first time, check the fingerprint against the University page before typing `yes`.

## 2. Get the code onto ISCA (once)

On the laptop, push first (`git push` in the project folder). Then on ISCA:

    git clone -b qwen35-thinking-off https://github.com/Arush70/Multi-Agent-Coding-Studio-with-trace-based-creativity-measurement.git macs
    cd macs

If the repository is private and the clone asks for a password, use a GitHub personal access
token as the password, or copy the code instead: on the laptop, in the project folder,

    git archive -o macs.zip HEAD
    scp macs.zip uid123@login.isca.ex.ac.uk:~

and on ISCA: `unzip -q macs.zip -d macs && cd macs`.

## 3. One-time setup (about 30 to 60 minutes)

Still on ISCA, inside the `macs` folder:

    srun --export ALL -N 1 -t 02:00:00 -p sq -A COMM514 --mem 16G -c 4 --pty -- bash
    bash isca/setup.sh
    exit

This installs Python and Ollama in your home folder, downloads Qwen3.5-27B (about 17 GB) and
UniXcoder, writes `config/models.yaml` with the pinned model revision, and runs the tests.
If your home folder is short of space (`quota -s`), put the tools elsewhere first, for example
`export MACS_PREFIX=/lustre/home/uid123/macs-tools`, and use the same setting for the jobs.

## 4. Start the main runs (27B, and 9B under the same design)

Design from 23 October 2026: 1 vs 6 learners, plus 4 learners for Tutor and Studio (11
conditions), 27 tasks x 6 seeds, and an 8192-token context or more (the jobs use 16384). Traces
from the earlier 2-learner design are archived first, never mixed in:

    python scripts/archive_traces.py --name 2026-10-23_two_learner_ctx4096 --apply
    sbatch isca/run_model.sbatch                     # Qwen3.5-27B, the main model
    PROFILE=local sbatch isca/run_model.sbatch       # Qwen3.5-9B, second full-design model

(`archive_traces.py` needs the conda environment: run
`source ~/macs-tools/miniforge3/etc/profile.d/conda.sh && conda activate macs` first.)

Each job builds the reference solutions for each task, runs all sessions (6 at a time on one
A100), then runs the open-problem search. It queues its own follow-up job for when the 24-hour
limit is reached, and continues where it stopped.

## 5. Check progress

    squeue -u $USER                      # is it queued (PD) or running (R)?
    tail -f data/jobs/isca-*.log         # what the job is doing (Ctrl+C to stop watching)
    ls data/traces | wc -l               # finished and running sessions so far

You can log out at any time; the job keeps running.

## 6. Bring the results back

When the log says `all done`, on the laptop in PowerShell, inside the project folder:

    scp uid123@login.isca.ex.ac.uk:~/macs/data/jobs/results-qwen35_27b.tgz .
    tar -xzf results-qwen35_27b.tgz

The traces carry the model name, so the dashboard and the analysis keep 27B and 9B apart.

## Bigger model: Qwen3.5-122B-A10B (scale check)

The largest Qwen3.5 that fits one ISCA node (about 70 GB at 4-bit, so it uses both A100s on the
node). It runs the reduced design, so the study has 9B, 27B and 122B from one model family.
Download it during setup by running `MACS_MODELS="qwen3.5:9b qwen3.5:27b qwen3.5:122b" bash isca/setup.sh`
(allow 3 hours for the srun job), then, after the 27B job has started:

    PROFILE=qwen35_122b TASKS=main CONDITIONS=A,B,C,D,D_minus,E,E_minus SEEDS=1-3 sbatch --gres=gpu:2 -c 16 --mem=128G isca/run_model.sbatch

## Large models from other families (cross-model panel)

gpt-oss 120B (OpenAI), Llama 4 Scout 109B (Meta) and Gemma 4 31B (Google). Download them with a
second setup run (inside an srun job, as in step 3; installed parts are skipped):

    MACS_MODELS="qwen3.5:9b qwen3.5:27b qwen3.5:122b gpt-oss:120b llama4:16x17b gemma4:31b" bash isca/setup.sh

Then submit one job per model (reduced design):

    D="TASKS=main CONDITIONS=A,B,C,D,D_minus,E,E_minus SEEDS=1-3"
    env $D PROFILE=gpt_oss_120b sbatch isca/run_model.sbatch
    env $D PROFILE=llama4_scout sbatch --gres=gpu:2 -c 16 --mem=128G isca/run_model.sbatch
    env $D PROFILE=gemma4_31b   sbatch isca/run_model.sbatch

DeepSeek is not in the panel: it is the independent AI judge in the rating study, and its full
models (about 400 GB) do not fit an ISCA node.

## Optional: more models on the GPU

gpt-oss-20b for the cross-model panel (reduced design):

    MACS_MODELS=gpt-oss:20b bash isca/setup.sh        # inside an srun job, as in step 3
    PROFILE=gpt_oss_20b TASKS=main CONDITIONS=A,B,C,D,D_minus,E,E_minus SEEDS=1-3 sbatch isca/run_model.sbatch
