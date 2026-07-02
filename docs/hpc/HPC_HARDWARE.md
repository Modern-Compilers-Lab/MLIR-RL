**The hardware specifics** for the three clusters : Bergamo, Jubail, and Dalma.

### 1. **Bergamo** (Bigmem partition on Jubail)
| Item              | Specification                          |
|-------------------|----------------------------------------|
| **Number of nodes** | 57                                    |
| **CPUs per node**   | **256**                               |
| **CPU Model**       | AMD EPYC 9754 (Genoa, Zen 4)          |
| **RAM per node**    | **1 TB**                              |
| **Use case**        | High-memory / bigmem jobs             |

This is currently the **largest single-node memory + core** option on the NYUAD HPC.

### 2. **Jubail** (Main / standard compute nodes)
| Item              | Specification                          |
|-------------------|----------------------------------------|
| **Number of nodes** | 233                                   |
| **CPUs per node**   | **128**                               |
| **CPU Model**       | AMD EPYC 7742 (Rome, Zen 2) @ 2.25 GHz |
| **RAM per node**    | **480 GB** (usable; 512 GB theoretical) |
| **Default**         | Most common partition (`compute`)     |

There is also **1 extra Bigmem Jubail node** (128 cores + 1 TB) separate from Bergamo.

### 3. **Dalma** (Older integrated cluster)
| Item              | Specification                                      |
|-------------------|----------------------------------------------------|
| **Number of nodes** | 432                                               |
| **CPUs per node**   | **28** or **40**                                  |
| **CPU Model**       | Mostly Intel Xeon E5-2680 v4<br>Some Xeon Gold 6148 (40 cores) |
| **RAM per node**    | Mostly **112 GB** (some 480 GB)                   |
| **Use case**        | Smaller / legacy jobs (often routed for small jobs) |

**Bigmem Dalma** (4 nodes): 32/64/72 cores + 1–2 TB RAM.

### Quick Comparison (CPU + RAM focus)

| Cluster / Type     | Nodes | Cores per Node | RAM per Node     | CPU Generation      |
|--------------------|-------|----------------|------------------|---------------------|
| **Bergamo**        | 57    | **256**        | **1 TB**         | AMD EPYC 9754 (newest) |
| **Jubail standard**| 233   | **128**        | **480 GB**       | AMD EPYC 7742       |
| **Dalma standard** | 432   | 28 / 40        | 112 GB (most)    | Intel Broadwell/Skylake |

**Tip for your jobs**:
- Need **massive RAM** (e.g. very large models) → use **Bergamo** (`--partition=bigmem` + high `--mem`).
- Balanced high-core jobs → Jubail standard nodes.
- Small/medium jobs → often land on Dalma nodes for faster scheduling.
