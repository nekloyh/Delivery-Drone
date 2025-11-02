# Evaluation script

The `eval_nav.py` script provides a basic framework for evaluating
trained policies on hold‑out maps.  It loads a PPO model, runs a
specified number of episodes and records metrics such as success,
collisions, time, energy and jerk.  The current implementation
produces a CSV with these values, but it leaves several tasks for the
user to complete:

1. **ATE/RPE integration.**  To compute Absolute Trajectory Error
   (ATE) and Relative Pose Error (RPE) as additional metrics you
   should modify the environment to log the ground truth and SLAM
   estimated poses at each step.  After each episode you can run the
   `evo_ate.sh` script to compute the ATE/RPE and append the results
   to the metrics dict.

2. **Logging format.**  Consider writing the episode metrics to a
   structured log file (CSV, JSON or Parquet) that matches the
   schema defined in `logging/schema.md`.  This will allow the
   energy metrics script and other analysis tools to consume the logs
   directly.

3. **Multiple agents.**  To evaluate baselines, wrap the baseline
   planners in a class that exposes a `predict` method similar to
   Stable‑Baselines3 models.  Then call `run_episode` with the
   baseline agent to collect metrics.

4. **Per‑step logging.**  If you wish to analyse energy or jerk at a
   finer temporal resolution you may log per‑step values inside
   `run_episode`.  For example, append the jerk at each step to a
   list and save it to disk.  This will enable CDF plots as shown in
   the report.

5. **Plotting.**  After evaluating all agents you can use libraries
   such as matplotlib or seaborn to create bar charts or CDF plots for
   the collected metrics.  Include such plots in your report to
   illustrate the performance differences between policies.
