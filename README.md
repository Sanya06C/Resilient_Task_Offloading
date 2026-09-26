# Resilient Task Offloading under Network Stress and Energy Variability in IIoT

This repository contains the implementation, experiments, and reproducibility artifacts for our research on **resilient task offloading in Industrial Internet of Things (IIoT) environments**.

The work builds upon the original **QECO** framework by Rahmati et al. and extends it in two directions:

- **Network-stress-aware task offloading** using real smart-factory traffic.
- **Energy-harvesting-aware task offloading** using real harvesting traces and battery state information.

The objective is to study whether energy-efficient task offloading remains beneficial when **network conditions degrade** and **device energy availability changes**.

---

## Research Motivation

Task offloading allows resource-constrained IoT devices to execute computation either:

- locally on the device,
- on Edge Server 0, or
- on Edge Server 1.

Although offloading can reduce local computation energy, it also introduces **transmission and waiting energy**.

Our literature study reviewed **51 papers**, including **31 core algorithmic studies**, and found that **77.4% of the core studies relied on simulation-only validation**.

This motivated us to study how an existing learned offloading policy behaves when exposed to conditions derived from **real industrial network traffic** and **real energy-harvesting measurements**.

---

# Base Framework - QECO

This work builds upon:

**QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing**

QECO combines:

- **Dueling Double Deep Q-Network (D3QN)** for offloading decisions.
- **Long Short-Term Memory (LSTM)** for capturing dynamic edge-server workloads.
- Three possible actions for each task:
  - Local execution
  - Edge Server 0
  - Edge Server 1

### Original QECO Resources

**Repository:**  
https://github.com/ImanRHT/QECO

**Paper:**  
I. Rahmati, H. Shah-Mansouri, and A. Movaghar,  
*"QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing,"*  
IEEE Transactions on Network Science and Engineering, 2025.

**DOI:**  
https://doi.org/10.1109/TNSE.2025.3556809

---

# Research Extensions

## 1. Real-Data-Driven Network Stress

The original QECO policy was kept **frozen** during this experiment.

The same pretrained checkpoints and workloads were used while only the effective communication condition was changed.

Real smart-factory traffic was processed to derive:

- Normal
- Low
- Medium
- High

network-stress conditions.

Traffic pressure was mapped to QECO's effective transmission capacity while keeping the remaining simulation parameters unchanged.

### Stress Profiles

| Stress Level | Load Multiplier `M` | Effective Capacity |
|:------------:|--------------------:|-------------------:|
| **Normal**   | 1.0000 | 14.0000 |
| **Low**      | 1.0000 | 14.0000 |
| **Medium**   | 2.7207 | 8.4877 |
| **High**     | 4.8964 | 6.3269 |

### Original QECO under Network Stress

| Stress Level | UE Energy (J) | Matched Offloading Saving | Deadline Violations |
|:------------:|--------------:|--------------------------:|--------------------:|
| **Normal**   | 434.376 | **+28.25%** | 7.47% |
| **Medium**   | 562.167 | **-8.54%** | 23.76% |
| **High**     | 614.073 | **-26.54%** | 36.90% |

### Observation

Under Normal conditions, offloading was energy-efficient.

At Medium stress, offloading became **8.54% more expensive** than executing the same tasks locally.

At High stress, the penalty increased to **26.54%**.

This identifies an **energy crossover**, showing that offloading is not always energy-efficient when communication conditions deteriorate.

---

## 2. Stress-Aware Selective Offloading

To improve decisions under degraded network conditions, we introduced a lightweight **stress-aware selector** on top of frozen QECO.

For every arriving task, the selector evaluates:

- Local execution
- Edge Server 0
- Edge Server 1

using:

- Task size
- Computation density
- Deadline
- Local queue state
- Transmission queue state
- Edge queue state
- Effective transmission capacity
- Estimated UE energy

Deadline-infeasible actions are rejected.

QECO's original decision is changed only when another feasible option is predicted to consume less UE energy.

### Original vs Stress-Aware QECO

| Stress | Method | UE Energy (J) | TX Energy (J) | Deadline Violations | Offload Ratio |
|:------:|:------|--------------:|--------------:|--------------------:|--------------:|
| **Normal** | Original QECO | 434.376 | 263.795 | 7.47% | 63.96% |
|  | Stress-Aware QECO | **397.618** | **246.729** | **6.84%** | 63.54% |
| **Medium** | Original QECO | 562.167 | 351.388 | 23.76% | 56.45% |
|  | Stress-Aware QECO | **522.685** | **246.872** | **12.97%** | 38.11% |
| **High** | Original QECO | 614.073 | 387.240 | 36.90% | 53.64% |
|  | Stress-Aware QECO | **567.089** | **241.910** | **24.99%** | 27.08% |

### Key Improvements

| Metric | Medium Stress | High Stress |
|:-------|--------------:|------------:|
| **UE Energy Reduction** | **7.02%** | **7.65%** |
| **TX Energy Reduction** | **29.74%** | **37.53%** |
| **Deadline Violation Reduction** | **45.39%** | **32.26%** |
| **Matched Saving** | -8.54% → **+4.91%** | -26.54% → **-21.75%** |

At Medium stress, the selector successfully restored positive matched energy savings.

At High stress, energy and deadline performance still improved, although matched offloading remained unfavorable under the configured power and transmission-capacity assumptions.

---

## 3. Energy-Harvesting Extension - EH-QECO v2

The second branch extends QECO for **energy-harvesting IoT devices**.

Two new features were added to the original QECO observation:

- **Battery State of Charge (SOC)**
- **Harvest availability**

The observation space therefore increased from **6 features to 8 features**.

Because the observation space changed, new D3QN models were **trained from scratch**.

### Training Setup

Five independent training seeds were used:

```text
101
202
303
404
505
```

The model was trained using a chronological training split and evaluated on an unseen final test split.

A total of **54,947 gradient updates** were recorded across the five training runs.

---

# Battery and Energy-Harvesting Model

EH-QECO v2 uses a **capacity-bounded Coulomb-counting battery model** with strict energy causality.

### Battery Parameters

| Parameter | Value |
|:----------|------:|
| Battery Capacity | **2000 mAh** |
| Nominal Voltage | **3.7 V** |
| Charge Efficiency | **0.95** |
| Discharge Efficiency | **0.95** |
| Simulation Step | **0.1 s** |

Energy causality ensures that an action cannot execute if the battery does not contain sufficient stored energy.

Each 60-second harvesting measurement is mapped to **600 consecutive 0.1-second QECO slots** using Zero-Order Hold.

---

# Energy-Harvesting Results

EH-aware policies were compared with matched EH-disabled runs.

| Network Stress | Δ UE Energy | Δ TX Energy | Δ Final SOC |
|:--------------:|------------:|------------:|------------:|
| **Normal** | **-0.502 J** | **-0.912 J** | ≈ **+0.055 pp** |
| **Medium** | **-0.999 J** | **-1.509 J** | ≈ **+0.055 pp** |
| **High** | **-1.420 J** | **-2.013 J** | ≈ **+0.055 pp** |

Negative energy differences indicate lower energy consumption compared with the matched EH-disabled baseline.

The primary benefit of EH-QECO was improved **energy usage and battery reserve**.

Short-horizon deadline-violation rates remained approximately similar, meaning the EH extension mainly improved energy management rather than service reliability.

---

# Long-Horizon Evaluation

Continuous evaluations were performed over longer physical time horizons.

| Duration | Main Observation |
|:--------:|:-----------------|
| **1 Hour** | No energy-induced task failures; minimum SOC remained **29.45%** |
| **6 Hours** | Battery first depleted after approximately **2.8 hours** |
| **24 Hours** | Harvesting partially offset consumption, but the continuous workload remained energy-deficit |

Energy harvesting therefore improves short-term battery reserve but does **not** make the tested continuously active workload indefinitely sustainable.

---

# Datasets

## Smart-Factory Network Traffic Dataset

Used to derive the real-data-driven network-stress profiles.

**Reference:**  
B. Brenner, J. Fabini, M. Offermanns, S. Semper, and T. Zseby,  
*"Malware Communication in Smart Factories: A Network Traffic Data Set,"*  
Computer Networks, 2024.

**DOI:**  
https://doi.org/10.1016/j.comnet.2024.110804

---

## UCLM Energy-Harvesting Dataset

Used for the battery, SOC, harvesting, and long-horizon experiments.

**Reference:**  
M. Kuzman, X. del Toro Garcia, S. Escolar, A. Caruso, S. Chessa, and J. C. Lopez,  
*"A Testbed and an Experimental Public Dataset for Energy-Harvested IoT Solutions,"*  
IEEE International Conference on Industrial Informatics (INDIN), 2019.

**DOI:**  
https://doi.org/10.1109/INDIN41052.2019.8972219

---

# Experimental Workflow

```text
                    Literature Synthesis
                            |
                            v
                     Validation Gap
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Network-Stress Branch       Energy-Harvesting Branch
              |                           |
              v                           v
      Smart-Factory Traffic          UCLM EH Dataset
              |                           |
              v                           v
       Frozen QECO Test          Battery + SOC Model
              |                           |
              v                           v
        Energy Crossover         D3QN Retraining
              |                           |
              v                           v
     Stress-Aware Selector      EH + Stress Evaluation
              |                           |
              v                           v
      Operating-Region          Sensitivity + Long-
          Analysis               Horizon Analysis
              |                           |
              +-------------+-------------+
                            |
                            v
              Resilient IIoT Offloading
                       Findings
```

---

# Repository Structure

```text
QECO/
│
├── main.py
├── MEC_Env.py
├── DDQN.py
├── Config.py
│
├── eh_extension_v2/
│   ├── battery dynamics
│   ├── energy-harvesting environment
│   ├── training utilities
│   └── evaluation outputs
│
├── notebooks/
│   ├── network stress analysis
│   ├── energy accounting
│   ├── EH time mapping
│   ├── battery dynamics
│   ├── core evaluation
│   └── sensitivity analysis
│
├── results/
│   ├── stress evaluation
│   ├── EH evaluation
│   └── sensitivity results
│
└── README.md
```

> The exact folder names may vary depending on the current repository version.

---

# Reproducibility

The experimental setup includes:

- Frozen QECO checkpoints for the network-stress branch
- Fixed workload replay
- Checkpoint and workload hashing
- Zero-learning verification during frozen-policy evaluation
- Five independent EH-QECO training seeds
- Chronological train/test split
- Paired EH-enabled and EH-disabled evaluations
- Sensitivity analysis
- 1-hour, 6-hour, and 24-hour continuous evaluations

---

# Main Findings

The experiments show that **task offloading is conditional rather than universally energy-efficient**.

Under good communication conditions, offloading reduces UE energy.

As communication service degrades, transmission and waiting energy increase and can make local execution more energy-efficient.

The stress-aware selector improves both energy consumption and deadline reliability under Medium and High network stress.

Energy harvesting further improves energy use and battery reserve, but it does not make the tested continuous high-duty workload indefinitely sustainable.

---

# Research Paper

**Resilient Task Offloading under Network Stress and Energy Variability in IIoT**

**Status:** Research manuscript / unpublished work

**Research Repository:**  
(https://github.com/Sanya06C/Task-Offloading)

---

# Attribution

This research builds upon the public **QECO** implementation developed by Rahmati et al.

Original QECO repository:

https://github.com/ImanRHT/QECO

The original QECO implementation and associated components remain attributed to their respective authors.

The following components were developed as part of the present research:

- Real-data-driven network-stress derivation
- Policy-preserving QECO stress evaluation
- Matched-task energy accounting
- Stress-aware selective offloading
- EH-QECO v2
- Battery and SOC modeling
- Energy-harvesting-aware D3QN retraining
- Sensitivity analysis
- Long-horizon evaluation

---

# Citation

If you use the original QECO implementation, please cite:

```bibtex
@article{rahmati2025qeco,
  title={QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing},
  author={Rahmati, Iman and Shah-Mansouri, Hamed and Movaghar, Ali},
  journal={IEEE Transactions on Network Science and Engineering},
  doi={10.1109/TNSE.2025.3556809},
  year={2025}
}
```

Citation information for the present research manuscript will be added after publication.

---

## License

This repository contains modifications and research extensions built upon the QECO project.

Please refer to the original QECO repository for the license governing the original implementation:

https://github.com/ImanRHT/QECO/blob/master/LICENSE
