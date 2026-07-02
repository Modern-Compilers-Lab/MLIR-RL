const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = '/scratch/mb10856/MLIR-RL';
const GRAPH_PATH = path.join(PROJECT_ROOT, '.understand-anything/intermediate/assembled-graph.json');
const LAYERS_PATH = path.join(PROJECT_ROOT, '.understand-anything/intermediate/layers.json');

function main() {
  console.log('Loading assembled-graph.json...');
  const graph = JSON.parse(fs.readFileSync(GRAPH_PATH, 'utf8'));
  
  // File-level node types
  const fileTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
  const fileNodes = graph.nodes.filter(n => fileTypes.has(n.type));
  
  console.log(`Found ${fileNodes.length} file-level nodes.`);
  
  // Layer definitions
  const layersMap = {
    'layer:rl-agent': {
      id: 'layer:rl-agent',
      name: 'Reinforcement Learning Agent',
      description: 'Reinforcement learning auto-scheduler core package, implementing agent models, actions, and environments.',
      nodeIds: []
    },
    'layer:data-utils': {
      id: 'layer:data-utils',
      name: 'Data Utilities',
      description: 'Data generation, extraction, and formatting utilities for loop nest benchmarks.',
      nodeIds: []
    },
    'layer:dashboard': {
      id: 'layer:dashboard',
      name: 'Streamlit Dashboard',
      description: 'Interactive Streamlit visualization dashboard for training progress and ablation analysis.',
      nodeIds: []
    },
    'layer:plots': {
      id: 'layer:plots',
      name: 'Plots and Metrics',
      description: 'Visualization scripts and dataset files for paper plotting.',
      nodeIds: []
    },
    'layer:manuscript': {
      id: 'layer:manuscript',
      name: 'Manuscript Source',
      description: 'LaTeX sources for the thesis or paper manuscript describing the methodology.',
      nodeIds: []
    },
    'layer:config': {
      id: 'layer:config',
      name: 'Configuration',
      description: 'Experiment and hyperparameter configuration JSON files.',
      nodeIds: []
    },
    'layer:documentation': {
      id: 'layer:documentation',
      name: 'Documentation',
      description: 'Project guides, tutorials, and architectural documentation.',
      nodeIds: []
    },
    'layer:infrastructure': {
      id: 'layer:infrastructure',
      name: 'Infrastructure & Scripts',
      description: 'Slurm execution scripts and helper scripts for running cluster workloads.',
      nodeIds: []
    },
    'layer:utilities': {
      id: 'layer:utilities',
      name: 'Utilities',
      description: 'Shared common utilities, logging, and singleton configuration managers.',
      nodeIds: []
    }
  };
  
  // Assign each file node to a layer based on path
  fileNodes.forEach(node => {
    const filePath = node.filePath;
    if (!filePath) {
      console.warn(`Node ${node.id} has no filePath! Skipping.`);
      return;
    }
    
    if (filePath.startsWith('rl_autoschedular/')) {
      layersMap['layer:rl-agent'].nodeIds.push(node.id);
    } else if (filePath.startsWith('data_utils/')) {
      layersMap['layer:data-utils'].nodeIds.push(node.id);
    } else if (filePath.startsWith('dashboard/')) {
      layersMap['layer:dashboard'].nodeIds.push(node.id);
    } else if (filePath.startsWith('plots/')) {
      layersMap['layer:plots'].nodeIds.push(node.id);
    } else if (filePath.startsWith('manuscript/')) {
      layersMap['layer:manuscript'].nodeIds.push(node.id);
    } else if (filePath.startsWith('config/')) {
      layersMap['layer:config'].nodeIds.push(node.id);
    } else if (filePath.startsWith('docs/')) {
      layersMap['layer:documentation'].nodeIds.push(node.id);
    } else if (filePath.startsWith('scripts/')) {
      layersMap['layer:infrastructure'].nodeIds.push(node.id);
    } else if (filePath.startsWith('utils/')) {
      layersMap['layer:utilities'].nodeIds.push(node.id);
    } else {
      // Root files
      if (node.type === 'document' || filePath.endsWith('.md')) {
        layersMap['layer:documentation'].nodeIds.push(node.id);
      } else if (node.type === 'config' || filePath.endsWith('.yml') || filePath.endsWith('.yaml') || filePath.endsWith('.txt') || filePath.startsWith('.')) {
        layersMap['layer:config'].nodeIds.push(node.id);
      } else {
        layersMap['layer:utilities'].nodeIds.push(node.id);
      }
    }
  });
  
  // Filter out empty layers
  const finalLayers = Object.values(layersMap).filter(l => l.nodeIds.length > 0);
  
  fs.writeFileSync(LAYERS_PATH, JSON.stringify(finalLayers, null, 2));
  console.log(`Successfully wrote ${finalLayers.length} layers to layers.json.`);
  finalLayers.forEach(l => {
    console.log(`  - ${l.name} (${l.id}): ${l.nodeIds.length} files`);
  });
}

main();
