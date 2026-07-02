const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PROJECT_ROOT = '/scratch/mb10856/MLIR-RL';
const INTERMEDIATE_DIR = path.join(PROJECT_ROOT, '.understand-anything/intermediate');

function main() {
  const commitHash = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
  const timestamp = new Date().toISOString();

  const assembledGraph = JSON.parse(fs.readFileSync(path.join(INTERMEDIATE_DIR, 'assembled-graph.json'), 'utf8'));
  const layers = JSON.parse(fs.readFileSync(path.join(INTERMEDIATE_DIR, 'layers.json'), 'utf8'));
  const tour = JSON.parse(fs.readFileSync(path.join(INTERMEDIATE_DIR, 'tour.json'), 'utf8'));

  const fullGraph = {
    version: "1.0.0",
    project: {
      name: "MLIR-RL",
      languages: ["cpp", "csv", "json", "markdown", "python", "shell", "tex", "txt"],
      frameworks: ["PyTorch", "Dask", "Neptune"],
      description: "Reinforcement-learning auto-scheduler for MLIR loop nests, optimizing schedules via PPO with process isolation and safety mechanisms. Note: this project has over 100 source files; consider scoping analysis to a subdirectory for faster results.",
      analyzedAt: timestamp,
      gitCommitHash: commitHash
    },
    nodes: assembledGraph.nodes,
    edges: assembledGraph.edges,
    layers: layers,
    tour: tour
  };

  const fullGraphPath = path.join(INTERMEDIATE_DIR, 'assembled-graph.json');
  fs.writeFileSync(fullGraphPath, JSON.stringify(fullGraph, null, 2));
  console.log('Successfully assembled full graph at', fullGraphPath);
}

main();
