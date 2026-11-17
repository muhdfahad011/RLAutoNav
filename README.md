# rlnav-carla: a hack-resistant RL navigation environment

A small RL environment for point-to-point driving, built on the CARLA API, designed around one question:
**how do you stop an agent from gaming the score?**

It has a CARLA backend for real runs and a kinematic fake backend, so tests, CI and cluster jobs need no simulator.

```
rlnav/
  types.py          State / Action / Task + geometry
  env.py            Gym-like NavEnv over any backend
  backend_carla.py  Real CARLA 0.9.13+ backend (vehicle, collision + lane-invasion sensors, sync mode)
  backend_fake.py   Kinematic bicycle model (used by tests/CI/K8s)
  reward.py         Dense, potential-based training reward
  verifier.py       Strict, independent pass/fail grader
  agent.py          LLM agent (Anthropic API) + heuristic + random baselines
  rollout.py        Parallel rollouts, every episode graded by the verifier
  evaluate.py       CLI
tests/              Reward-hack tests (each test = one exploit)
harbor_task/        Harbor-style task: instruction, Dockerfile, hidden-seed grader, reference solution
k8s/job.yaml        Indexed Job for parallel rollouts
Dockerfile          Reproducible runner
```

## Quick start
```bash
pip install -e ".[dev]"
pytest -q                                              # 18 tests, no simulator
python -m rlnav.evaluate --policy heuristic            # pass_rate 1.0
python -m rlnav.evaluate --policy random               # pass_rate 0.0
ANTHROPIC_API_KEY=... python -m rlnav.evaluate --policy llm --episodes 5
python -m rlnav.evaluate --backend carla --policy heuristic   # needs a CARLA server
docker build -t rlnav . && docker run rlnav
./scripts/sync_harbor.sh                               # prepare harbor_task build context
kubectl apply -f k8s/job.yaml
```

## Reward hacking: what I defended against
My earlier project (RLAutoNav) rewarded speed, which an agent can farm by driving fast in circles.
Here the reward and the grader are separate, and every verifier check maps to a concrete exploit:

| Exploit | Defence | Test |
|---|---|---|
| Drive fast/circle to farm speed reward | Reward is potential-based on distance-to-goal (loops sum to <= 0); no speed term | `test_circling_earns_no_reward`, `test_speed_alone_earns_no_reward` |
| Circle forever | `goal_not_reached` | `test_circling_is_rejected` |
| Loop, then finish | `excessive_turning`, `inefficient_path` | `test_loops_then_goal_rejected` |
| Teleport to goal | Per-step displacement bound | `test_teleport_rejected` |
| Speeding | `speed_limit` check | `test_speed_farming_rejected` |
| Stand still | Must reach goal | `test_standing_still_rejected` |
| Fabricated start / timestamps | `bad_start`, `bad_timestamps` | `test_fabricated_start_rejected`, `test_faked_timestamps_rejected` |
| Hit things | `collision` check | `test_collision_rejected` |

Harbor grading also runs the agent's policy inside the trusted env on **hidden seeds**, so the agent never
supplies the trajectory it is judged on. Broken policy code or API failures degrade to braking instead of crashing the grader.

## Honest limitations
- `backend_carla.py` follows the CARLA 0.9.13+ API but is not exercised by the tests (no server in CI).
- Verifier thresholds (turn limit, path ratio) are tuned for this simple task and would need re-tuning in real towns with road networks.
- The Harbor layout follows the commonly documented task format (`instruction.md`, `task.toml`, `environment/`, `tests/test.sh`, `solution/`); check it against the current Harbor docs.
- Next steps: waypoint/route-based verification in CARLA, SFT/RLVR on the verifier signal, sandboxing the agent-written policy (gVisor/nsjail).
