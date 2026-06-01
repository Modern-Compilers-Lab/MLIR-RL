# LLM Pipeline Generation Metrics — Action Spaces v48–v52

3-layer Claude Code pipeline (**enumeration → implementation → exploration**) for five action spaces: v48 matmul, v49 conv2d, v50 pooling, v51 add, v52 relu. Each stage = one `claude` session (model claude-opus-4-6). Reported metrics: **tokens**, **time**, **tool calls**.

- **input (prompt)** — tokens in the initial instruction prompt for the stage
- **output (generated)** — output tokens produced across all turns (exact, from transcripts).
- **active time** — model run time.
- **tool calls** — total `tool_use` invocations across the session and its subagents.

## 1. Per stage × version

| Version | Kernel | Stage | Input (prompt) | Output (generated) | Active time | Tool calls |
|---|---|---|--:|--:|--:|--:|
| v48 | matmul | enumeration | 1,192 | 5,826 | 4m36s | 15 |
| v48 | matmul | implementation | 1,556 | 13,273 | 8m46s | 86 |
| v48 | matmul | exploration | 1,967 | 65,327 | 29m32s | 235 |
| v49 | conv2d | enumeration | 1,836 | 9,536 | 7m19s | 13 |
| v49 | conv2d | implementation | 1,947 | 28,644 | 19m13s | 95 |
| v49 | conv2d | exploration | 2,526 | 101,548 | 1h57m | 680 |
| v50 | pooling | enumeration | 960 | 2,193 | 1m55s | 13 |
| v50 | pooling | implementation | 1,491 | 24,601 | 14m59s | 103 |
| v50 | pooling | exploration | 1,990 | 48,471 | 38m11s | 251 |
| v51 | add | enumeration | 1,279 | 5,787 | 2m39s | 10 |
| v51 | add | implementation | 1,645 | 42,635 | 14m46s | 89 |
| v51 | add | exploration | 2,054 | 51,066 | 37m35s | 322 |
| v52 | relu | enumeration | 1,445 | 2,190 | 3m09s | 12 |
| v52 | relu | implementation | 1,818 | 32,133 | 24m13s | 114 |
| v52 | relu | exploration | 2,215 | 47,733 | 26m39s | 196 |

## 2. Per-version totals

| Version | Kernel | Input (prompt) | Output (generated) | Active time | Tool calls |
|---|---|--:|--:|--:|--:|
| v48 | matmul | 4,715 | 84,426 | 42m54s | 336 |
| v49 | conv2d | 6,309 | 139,728 | 2h23m | 788 |
| v50 | pooling | 4,441 | 75,265 | 55m04s | 367 |
| v51 | add | 4,978 | 99,488 | 55m00s | 421 |
| v52 | relu | 5,478 | 82,056 | 54m00s | 322 |

## 3. Per-stage totals

| Stage | Input (prompt) | Output (generated) | Active time | Tool calls |
|---|--:|--:|--:|--:|
| enumeration | 6,712 | 25,532 | 19m37s | 63 |
| implementation | 8,457 | 141,286 | 1h21m | 487 |
| exploration | 10,752 | 314,145 | 4h09m | 1684 |

## 4. Action-space tool calls per benchmark family (complete MCP action set)

Every action tool exposed by each `rl-action-v*` MCP server, with call counts. Action tools are invoked only during exploration.

| Family | Action tool | Calls |
|---|---|--:|
| matmul | `tiling_tool` | 24 |
| matmul | `loop_interchange_tool` | 15 |
| matmul | `vectorization_sequential_tool` | 15 |
| matmul | `vectorization_parallel_tool` | 40 |
| matmul | `parallelization_tiling_tool` | 15 |
| matmul | `parallelization_threads_tool` | 11 |
| conv2d | `tiling_tool` | 91 |
| conv2d | `loop_interchange_tool` | 43 |
| conv2d | `promotion_tool` | 42 |
| conv2d | `im2col_lowering_tool` | 32 |
| conv2d | `vectorization_seq_tool` | 21 |
| conv2d | `vectorization_par_tool` | 7 |
| conv2d | `parallelization_tile_tool` | 54 |
| conv2d | `parallelization_threads_tool` | 40 |
| pooling | `tiling_tool` | 26 |
| pooling | `loop_interchange_tool` | 11 |
| pooling | `promotion_tool` | 5 |
| pooling | `sequential_vectorization_tool` | 31 |
| pooling | `parallel_vectorization_tool` | 15 |
| pooling | `tiling_parallelization_tool` | 18 |
| pooling | `thread_parallelization_tool` | 12 |
| add | `tiling_tool` | 14 |
| add | `loop_interchange_tool` | 11 |
| add | `vectorization_seq_tool` | 23 |
| add | `vectorization_par_tool` | 58 |
| add | `parallelization_tile_tool` | 26 |
| add | `parallelization_threads_tool` | 22 |
| relu | `tiling_tool` | 20 |
| relu | `loop_interchange_tool` | 10 |
| relu | `vectorization_sequential_tool` | 18 |
| relu | `vectorization_parallel_tool` | 30 |
| relu | `parallelization_tiling_tool` | 10 |
| relu | `parallelization_threads_tool` | 10 |

Action tools per family: matmul 6 (120 calls) · conv2d 8 (330 calls) · pooling 7 (118 calls) · add 6 (154 calls) · relu 6 (98 calls).

## 5. Other tool calls — MLIR-tools MCP + built-in (by stage)

| Tool | Total | enumeration | implementation | exploration |
|---|--:|--:|--:|--:|
| `mcp__mlir-tools__execute_mlir_code` | 410 | 0 | 24 | 386 |
| `Read` | 213 | 20 | 100 | 93 |
| `mcp__mlir-tools__measure_speedup` | 183 | 0 | 3 | 180 |
| `Write` | 121 | 12 | 99 | 10 |
| `TaskUpdate` | 120 | 0 | 44 | 76 |
| `Bash` | 81 | 16 | 60 | 5 |
| `TaskCreate` | 65 | 0 | 26 | 39 |
| `mcp__mlir-tools__transform_mlir_code` | 53 | 0 | 53 | 0 |
| `Edit` | 47 | 4 | 23 | 20 |
| `Glob` | 40 | 11 | 29 | 0 |
| `Task` | 23 | 0 | 1 | 22 |
| `mcp__mlir-tools__delegate_documentation_lookup` | 22 | 0 | 22 | 0 |
| `TaskOutput` | 9 | 0 | 0 | 9 |
| `mcp__mlir-tools__execute_torch_conv2d_by_shape` | 6 | 0 | 0 | 6 |
| `Grep` | 5 | 0 | 1 | 4 |
| `mcp__mlir-tools__execute_torch_pooling_nchw_max_by_shape` | 4 | 0 | 0 | 4 |
| `mcp__mlir-tools__execute_torch_add_by_shape` | 4 | 0 | 1 | 3 |
| `mcp__mlir-tools__execute_torch_relu_by_shape` | 4 | 0 | 1 | 3 |
| `mcp__mlir-tools__execute_torch_matmul_by_shape` | 3 | 0 | 0 | 3 |
| `TaskList` | 1 | 0 | 0 | 1 |

## 6. Other tool calls — MLIR-tools MCP + built-in (by benchmark family)

| Tool | Total | matmul | conv2d | pooling | add | relu |
|---|--:|--:|--:|--:|--:|--:|
| `mcp__mlir-tools__execute_mlir_code` | 410 | 50 | 162 | 61 | 78 | 59 |
| `Read` | 213 | 32 | 68 | 39 | 37 | 37 |
| `mcp__mlir-tools__measure_speedup` | 183 | 36 | 26 | 40 | 54 | 27 |
| `Write` | 121 | 24 | 29 | 25 | 21 | 22 |
| `TaskUpdate` | 120 | 24 | 58 | 22 | 8 | 8 |
| `Bash` | 81 | 11 | 18 | 19 | 18 | 15 |
| `TaskCreate` | 65 | 12 | 31 | 11 | 7 | 4 |
| `mcp__mlir-tools__transform_mlir_code` | 53 | 6 | 9 | 9 | 8 | 21 |
| `Edit` | 47 | 9 | 21 | 2 | 8 | 7 |
| `Glob` | 40 | 6 | 9 | 9 | 6 | 10 |
| `Task` | 23 | 3 | 10 | 3 | 3 | 4 |
| `mcp__mlir-tools__delegate_documentation_lookup` | 22 | 0 | 6 | 5 | 5 | 6 |
| `TaskOutput` | 9 | 0 | 0 | 0 | 9 | 0 |
| `mcp__mlir-tools__execute_torch_conv2d_by_shape` | 6 | 0 | 6 | 0 | 0 | 0 |
| `Grep` | 5 | 0 | 4 | 0 | 1 | 0 |
| `mcp__mlir-tools__execute_torch_pooling_nchw_max_by_shape` | 4 | 0 | 0 | 4 | 0 | 0 |
| `mcp__mlir-tools__execute_torch_add_by_shape` | 4 | 0 | 0 | 0 | 4 | 0 |
| `mcp__mlir-tools__execute_torch_relu_by_shape` | 4 | 0 | 0 | 0 | 0 | 4 |
| `mcp__mlir-tools__execute_torch_matmul_by_shape` | 3 | 3 | 0 | 0 | 0 | 0 |
| `TaskList` | 1 | 0 | 1 | 0 | 0 | 0 |
