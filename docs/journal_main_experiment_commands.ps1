param(
    [switch]$RunTrain,
    [switch]$RunAnalyze,
    [switch]$RunMainline,
    [switch]$RunMechanism,
    [switch]$Execute,
    [string]$Device = "cuda",
    [int[]]$Seeds = @(42, 52, 62),
    [int]$NumShards = 8,
    [string]$RepresentativeRun = "",
    [string]$RepresentativeCheckpoint = "selected",
    [string]$OptimizedSchedule = ""
)

$ErrorActionPreference = "Stop"

$ROOT = "C:\Users\29341\Desktop\code_0420\lettuce_control_drl_new_0422"
$PY = "C:\Users\29341\.conda\envs\control_env\python.exe"
$DefaultSchedule = "t1=14|t2=14|N1=20|rho2=36"

Set-Location $ROOT

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    $line = ($Command | ForEach-Object {
        if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
    }) -join " "
    Write-Host "[CMD] $line"

    if ($Execute) {
        & $Command[0] @($Command[1..($Command.Length - 1)])
    }
}

function Require-RepresentativeRun {
    if ([string]::IsNullOrWhiteSpace($RepresentativeRun)) {
        throw "RepresentativeRun is required for mainline/mechanism steps. Set -RepresentativeRun after seed analysis."
    }
}

function Resolve-OptimizedSchedule {
    if (-not [string]::IsNullOrWhiteSpace($OptimizedSchedule)) {
        return $OptimizedSchedule
    }
    $jsonPath = Join-Path $ROOT "results/journal_main/exp05_synergy_matrix/synergy_matrix_summary.json"
    if (-not (Test-Path $jsonPath)) {
        throw "Optimized schedule is not set and synergy_matrix_summary.json was not found. Run M5 first or pass -OptimizedSchedule."
    }
    $payload = Get-Content $jsonPath -Raw | ConvertFrom-Json
    $key = [string]$payload.optimized_schedule_key
    if ([string]::IsNullOrWhiteSpace($key)) {
        throw "optimized_schedule_key is missing in synergy_matrix_summary.json."
    }
    return $key
}

if (-not ($RunTrain -or $RunAnalyze -or $RunMainline -or $RunMechanism)) {
    Write-Host "Select at least one stage:"
    Write-Host "  -RunTrain"
    Write-Host "  -RunAnalyze"
    Write-Host "  -RunMainline"
    Write-Host "  -RunMechanism"
    Write-Host ""
    Write-Host "Append -Execute to actually run. Without -Execute this script only prints commands."
    exit 0
}

if ($RunTrain) {
    foreach ($seed in $Seeds) {
        $exp = "journal_main_respid_sac_e500_s${seed}_v1"
        Invoke-Step @(
            $PY,
            "experiments/train_pfal_contextual_128_500.py",
            "--experiment", $exp,
            "--seed", "$seed",
            "--device", $Device,
            "--no_wandb"
        )
    }
}

if ($RunAnalyze) {
    $experiments = @()
    foreach ($seed in $Seeds) {
        $experiments += "journal_main_respid_sac_e500_s${seed}_v1"
    }

    $analyzeCmd = @(
        $PY,
        "experiments/analyze_rl_runs.py",
        "--experiments"
    ) + $experiments + @(
        "--out_dir", "results/journal_main/seed_analysis"
    )
    Invoke-Step $analyzeCmd

    foreach ($exp in $experiments) {
        $suffix = $exp.Split("_")[-2]
        Invoke-Step @(
            $PY,
            "experiments/compare_rl_checkpoints.py",
            "--load", $exp,
            "--device", $Device,
            "--checkpoints", "best", "selected", "final", "auto",
            "--n_eval_schedules", "32",
            "--n_eval_episodes_per_schedule", "5",
            "--eval_selection", "reference_stratified",
            "--out_dir", "results/journal_main/checkpoint_compare_${suffix}"
        )
    }
}

if ($RunMainline) {
    Require-RepresentativeRun

    Invoke-Step @(
        $PY,
        "experiments/1_mainline_system_validation.py",
        "--controllers", "pid", "rl",
        "--load", $RepresentativeRun,
        "--device", $Device,
        "--schedule", "t1=14,t2=14,N1=20,rho2=36",
        "--out-dir", "results/journal_main/exp01_system_validation"
    )

    for ($i = 0; $i -lt $NumShards; $i++) {
        Invoke-Step @(
            $PY,
            "experiments/2_mainline_exact_pid_baseline.py",
            "--num-shards", "$NumShards",
            "--shard-id", "$i",
            "--out-dir", "results/journal_main/exp02_exact_pid_baseline"
        )
    }

    Invoke-Step @(
        $PY,
        "experiments/2_mainline_exact_pid_baseline.py",
        "--num-shards", "$NumShards",
        "--merge-shards-only",
        "--out-dir", "results/journal_main/exp02_exact_pid_baseline"
    )

    for ($i = 0; $i -lt $NumShards; $i++) {
        Invoke-Step @(
            $PY,
            "experiments/3_mainline_exact_rl_baseline.py",
            "--load", $RepresentativeRun,
            "--load-checkpoint", $RepresentativeCheckpoint,
            "--device", $Device,
            "--num-shards", "$NumShards",
            "--shard-id", "$i",
            "--out-dir", "results/journal_main/exp03_exact_rl_baseline"
        )
    }

    Invoke-Step @(
        $PY,
        "experiments/3_mainline_exact_rl_baseline.py",
        "--load", $RepresentativeRun,
        "--load-checkpoint", $RepresentativeCheckpoint,
        "--device", $Device,
        "--num-shards", "$NumShards",
        "--merge-shards-only",
        "--out-dir", "results/journal_main/exp03_exact_rl_baseline"
    )

    Invoke-Step @(
        $PY,
        "experiments/4_mainline_pid_rl_comparison.py",
        "--pid-csv", "results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv",
        "--rl-csv", "results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv",
        "--out-dir", "results/journal_main/exp04_pid_rl_comparison"
    )

    Invoke-Step @(
        $PY,
        "experiments/5_mainline_synergy_matrix.py",
        "--pid-csv", "results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv",
        "--rl-csv", "results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv",
        "--out-dir", "results/journal_main/exp05_synergy_matrix"
    )
}

if ($RunMechanism) {
    Require-RepresentativeRun
    $optKey = Resolve-OptimizedSchedule

    Invoke-Step @(
        $PY,
        "experiments/2_mainline_exact_pid_baseline.py",
        "--schedule-key", $DefaultSchedule,
        "--save-detailed-traces",
        "--save-batch-trajectories",
        "--out-dir", "results/journal_main/mechanism/pid_default"
    )

    Invoke-Step @(
        $PY,
        "experiments/3_mainline_exact_rl_baseline.py",
        "--load", $RepresentativeRun,
        "--load-checkpoint", $RepresentativeCheckpoint,
        "--device", $Device,
        "--schedule-key", $DefaultSchedule,
        "--save-detailed-traces",
        "--save-batch-trajectories",
        "--out-dir", "results/journal_main/mechanism/rl_default"
    )

    Invoke-Step @(
        $PY,
        "experiments/2_mainline_exact_pid_baseline.py",
        "--schedule-key", $optKey,
        "--save-detailed-traces",
        "--save-batch-trajectories",
        "--out-dir", "results/journal_main/mechanism/pid_optimized"
    )

    Invoke-Step @(
        $PY,
        "experiments/3_mainline_exact_rl_baseline.py",
        "--load", $RepresentativeRun,
        "--load-checkpoint", $RepresentativeCheckpoint,
        "--device", $Device,
        "--schedule-key", $optKey,
        "--save-detailed-traces",
        "--save-batch-trajectories",
        "--out-dir", "results/journal_main/mechanism/rl_optimized"
    )

    Invoke-Step @(
        $PY,
        "experiments/analyze_rl_gain_mechanisms.py",
        "--pid-csv", "results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv",
        "--rl-csv", "results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv",
        "--pid-trace-csv", "results/journal_main/mechanism/pid_default/detailed_traces/t1-14__t2-14__N1-20__rho2-36.csv",
        "--rl-trace-csv", "results/journal_main/mechanism/rl_default/detailed_traces/t1-14__t2-14__N1-20__rho2-36.csv",
        "--out-dir", "results/journal_main/exp06_gain_mechanism_default"
    )
}
