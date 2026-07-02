const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = '/scratch/mb10856/MLIR-RL';
const GRAPH_PATH = path.join(PROJECT_ROOT, '.understand-anything/intermediate/assembled-graph.json');
const TOUR_PATH = path.join(PROJECT_ROOT, '.understand-anything/intermediate/tour.json');

function main() {
  console.log('Loading assembled-graph.json...');
  const graph = JSON.parse(fs.readFileSync(GRAPH_PATH, 'utf8'));
  const nodeIds = new Set(graph.nodes.map(n => n.id));
  
  console.log(`Loaded ${nodeIds.size} node IDs from graph.`);
  
  // Design the tour steps
  const tour = [
    {
      order: 1,
      title: 'Project Overview & Setup',
      description: 'Start with the project documentation. README.md covers the prerequisite compilers, setup steps, and environment variables needed to get MLIR-RL running, while the configuration docs explain how to configure loop nest scheduling runs.',
      nodeIds: ['document:README.md', 'document:config/README.md']
    },
    {
      order: 2,
      title: 'Training Entry Point',
      description: 'The main training script initializes the environment, collects trajectories from loop scheduler runs, and updates the neural networks using Proximal Policy Optimization (PPO).',
      nodeIds: ['file:rl_autoschedular/rl_autoschedular_paper/train.py', 'file:rl_autoschedular/rl_autoschedular_paper/ppo.py']
    },
    {
      order: 3,
      title: 'RL Scheduling Environment',
      description: 'The Gym-style reinforcement learning environment wraps the MLIR loop nests, tracking current state features, valid transformations, and calculating optimization speedup rewards.',
      nodeIds: ['file:rl_autoschedular/rl_autoschedular_paper/env.py', 'file:rl_autoschedular/rl_autoschedular_paper/state.py']
    },
    {
      order: 4,
      title: 'Loop Transformations',
      description: 'Applies loop transformations like tiling, loop interchange, parallelization, and vectorization to the MLIR loop nests. It invokes Python bindings or CLI fallbacks dynamically.',
      nodeIds: ['file:rl_autoschedular/rl_autoschedular_paper/transforms.py', 'file:rl_autoschedular/rl_autoschedular_paper/actions/base.py']
    },
    {
      order: 5,
      title: 'Neural Network Models',
      description: 'Defines the actor-critic policy model and loop nest representation networks, encoding loop nest features into sequence embeddings via LSTM or self-attention layers.',
      nodeIds: ['file:rl_autoschedular/rl_autoschedular_paper/model.py', 'file:rl_autoschedular/rl_autoschedular_paper/observation.py']
    },
    {
      order: 6,
      title: 'Timing Execution & Safety',
      description: 'Measures benchmark execution times inside a process-isolated sandbox to isolate MLIR JIT compiler crashes and prevent training corruption.',
      nodeIds: ['file:rl_autoschedular/rl_autoschedular_paper/execution.py', 'file:rl_autoschedular/rl_autoschedular_paper/utils/bindings_process.py'],
      languageLesson: 'Using multiprocessing and OS signal handlers (such as SIGABRT) protects the training process from JIT compiler crashes or illegal memory access during code compilation.'
    },
    {
      order: 7,
      title: 'Slurm HPC Scripting',
      description: 'Orchestrates cluster execution by setting up Conda environments, exporting variables, and launching training or evaluation runs in batch queues.',
      nodeIds: ['file:scripts/train/train.sh', 'file:scripts/eval/eval.sh'],
      languageLesson: 'Shell scripts handle background environment setups dynamically. Caching dependencies and libraries in cluster paths prevents duplicate builds across multiple jobs.'
    }
  ];
  
  // Validate that all nodeIds exist
  let valid = true;
  tour.forEach(step => {
    step.nodeIds.forEach(id => {
      if (!nodeIds.has(id)) {
        console.error(`Error: Node ID ${id} in step "${step.title}" does not exist in the graph!`);
        valid = false;
      }
    });
  });
  
  if (!valid) {
    console.error('Validation failed. Aborting.');
    process.exit(1);
  }
  
  fs.writeFileSync(TOUR_PATH, JSON.stringify(tour, null, 2));
  console.log(`Successfully wrote ${tour.length} steps to tour.json.`);
}

main();
