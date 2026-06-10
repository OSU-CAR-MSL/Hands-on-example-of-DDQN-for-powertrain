# Hands-on-example-of-DDQN-for-powertrain
Hands on example of DDQN for series hybrid powertrain. It uses vanilla single head DDQN to find the optimal powersplit between engine and battery.

# What the notebook does (summary)
It's a Double DQN (DDQN) reinforcement-learning energy-management strategy for a hybrid electric powertrain. The agent decides the battery/engine power split at each second of a drive cycle to minimize fuel while respecting SOC and power limits.

State: [SOC, P_dem]
Action: discrete battery power P_batt
Reward: -Kf·fuel_rate·Δt − Ks·(SOC−0.55)²
Net: MLP 2→256→256→128→action_dim
Notable mechanics: physically-valid action masking (get_Pbatt_bounds), an 8th-order polynomial fuel map (flowrate), experience replay, target network, gradient clipping, periodic checkpointing.
