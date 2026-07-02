const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PROJECT_ROOT = '/scratch/mb10856/MLIR-RL';
const SKILL_DIR = '/home/mb10856/.understand-anything/repo/understand-anything-plugin/skills/understand';
const BATCHES_JSON_PATH = path.join(PROJECT_ROOT, '.understand-anything/intermediate/batches.json');

// Helper to determine node type and prefix based on path and category
function getNodeTypeAndPrefix(filePath, category) {
  const base = path.basename(filePath);
  const ext = path.extname(filePath).toLowerCase();
  
  if (category === 'code') return { type: 'file', prefix: 'file:' };
  if (category === 'config') return { type: 'config', prefix: 'config:' };
  if (category === 'docs') return { type: 'document', prefix: 'document:' };
  
  if (category === 'infra') {
    if (base === 'Dockerfile' || base.startsWith('Dockerfile.') || base.startsWith('docker-compose.') || base === 'compose.yml' || base === 'compose.yaml') {
      return { type: 'service', prefix: 'service:' };
    }
    if (filePath.includes('.github/workflows/') || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') {
      return { type: 'pipeline', prefix: 'pipeline:' };
    }
    return { type: 'resource', prefix: 'resource:' };
  }
  
  if (category === 'data') {
    if (ext === '.sql') return { type: 'table', prefix: 'table:' };
    if (ext === '.graphql' || ext === '.gql' || ext === '.proto' || ext === '.prisma') {
      return { type: 'schema', prefix: 'schema:' };
    }
    return { type: 'endpoint', prefix: 'endpoint:' };
  }
  
  if (category === 'script') return { type: 'file', prefix: 'file:' };
  if (category === 'markup') return { type: 'file', prefix: 'file:' };
  
  return { type: 'file', prefix: 'file:' };
}

// Generate tags based on filename, category and path
function generateTags(filePath, category, exportsCount) {
  const base = path.basename(filePath).toLowerCase();
  const ext = path.extname(filePath).toLowerCase();
  
  const tags = [];
  
  if (base.includes('test') || base.includes('spec')) {
    tags.push('test', 'quality-assurance');
  }
  
  if (category === 'config') {
    tags.push('configuration', 'project-setup');
    if (ext === '.json') tags.push('json-config');
    if (ext === '.yml' || ext === '.yaml') tags.push('yaml-config');
  } else if (category === 'docs') {
    tags.push('documentation', 'reference');
    if (ext === '.md') tags.push('markdown');
    if (ext === '.tex') tags.push('latex', 'manuscript');
  } else if (category === 'infra') {
    tags.push('infrastructure', 'deployment');
    if (base.includes('docker')) tags.push('containerization', 'docker');
    if (base.includes('workflow') || base.includes('ci')) tags.push('ci-cd', 'automation');
  } else if (category === 'data') {
    tags.push('data-assets', 'schema');
    if (ext === '.csv') tags.push('dataset', 'csv-data');
    if (ext === '.sql') tags.push('database', 'sql-queries');
  } else if (category === 'script') {
    tags.push('scripting', 'automation');
    if (ext === '.sh') tags.push('shell-script', 'bash');
  } else {
    tags.push('source-code');
    if (ext === '.py') tags.push('python');
    if (ext === '.cpp') tags.push('cpp', 'native');
    
    // Core structural roles in MLIR-RL
    if (filePath.includes('rl_autoschedular')) {
      tags.push('autoscheduler', 'reinforcement-learning');
      if (filePath.includes('actions/')) tags.push('rl-action', 'loop-transformation');
      if (filePath.includes('utils/')) tags.push('rl-utility');
      if (base === 'ppo.py') tags.push('ppo-algorithm', 'policy-gradient');
      if (base === 'model.py') tags.push('neural-network', 'embeddings');
      if (base === 'env.py') tags.push('rl-environment', 'mlir-bindings');
      if (base === 'train.py') tags.push('entry-point', 'training-loop');
      if (base === 'evaluate.py') tags.push('entry-point', 'evaluation-harness');
    }
    if (filePath.includes('data_utils')) {
      tags.push('data-processing', 'mlir-extraction');
    }
  }
  
  if (exportsCount > 10) {
    tags.push('barrel-module');
  }
  
  // Ensure we have at least 3 and at most 5 tags
  while (tags.length < 3) {
    tags.push('mlir-rl');
  }
  return tags.slice(0, 5);
}

// Generate file summary dynamically
function generateSummary(filePath, category, fileInfo) {
  const base = path.basename(filePath);
  
  if (category === 'config') {
    return `Configuration settings for ${base} controlling runtime or build parameters.`;
  }
  if (category === 'docs') {
    return `Documentation file containing instructions, guides, or specifications for ${base}.`;
  }
  if (category === 'infra') {
    return `Infrastructure setup for ${base} orchestrating container builds, cluster environments, or pipelines.`;
  }
  if (category === 'data') {
    return `Data assets containing offline datasets, metrics, or schema definitions in ${base}.`;
  }
  if (category === 'script') {
    return `Script file executing automated training, evaluation, or setup scripts in the workspace.`;
  }
  
  // Custom python file summaries based on their names
  if (filePath.includes('rl_autoschedular')) {
    if (base === 'ppo.py') {
      return "Implements the Proximal Policy Optimization (PPO) reinforcement learning update loop and updates actor-critic policies.";
    }
    if (base === 'model.py') {
      return "Defines the actor-critic neural network architectures, including LSTM/Transformer-based loop nest encoders.";
    }
    if (base === 'env.py') {
      return "Defines the reinforcement learning scheduling environment wrapping loop states, schedules, and reward signals.";
    }
    if (base === 'state.py') {
      return "Represents loop nest characteristics, variables, and loops, extracting MLIR block features.";
    }
    if (base === 'execution.py') {
      return "Manages compilation, timing execution of MLIR modules under process isolation with timeout safety.";
    }
    if (base === 'transforms.py') {
      return "Implements MLIR loop transformations (interchange, tiling, parallelization, vectorization) using python bindings or CLI fallbacks.";
    }
    if (base === 'train.py') {
      return "Main training entry point that launches the reinforcement learning auto-scheduling loops on code benchmarks.";
    }
    if (base === 'evaluate.py') {
      return "Evaluation script to assess auto-scheduled policies on test benchmarks against heuristic compile times.";
    }
    if (filePath.includes('actions/')) {
      return `Implements the action logic and checks for loop transformation: ${base.split('.')[0]}.`;
    }
    return `Autoscheduling core python module defining policies, state mappings, or utilities.`;
  }
  
  if (filePath.includes('data_utils')) {
    if (base === 'extract_blocks.py') {
      return "Extracts nested loop blocks from large MLIR functions and patches them for isolated timing.";
    }
    if (base === 'extract_ops.py') {
      return "Extracts detailed Linalg operation signatures, loop types, and memory access patterns from MLIR.";
    }
    if (base === 'orchestrate.py') {
      return "Orchestrates datasets creation pipeline: converting vision/transformer models to MLIR and timing them.";
    }
    return `Data processing tool assisting in MLIR loop feature extraction and data preparation.`;
  }
  
  if (filePath.includes('dashboard')) {
    return `Streamlit web dashboard component displaying evaluation runs, training metrics, or ablation plots.`;
  }
  
  return `Python module contributing to the MLIR RL loop auto-scheduling system.`;
}

function main() {
  console.log('Loading batches.json...');
  const batchesData = JSON.parse(fs.readFileSync(BATCHES_JSON_PATH, 'utf8'));
  const totalBatches = batchesData.totalBatches;
  const batches = batchesData.batches;
  
  console.log(`Starting Phase 2 analysis: processing ${totalBatches} batches...`);
  
  // Cache target prefixes across the codebase to resolve import edge target IDs
  const pathPrefixMap = {};
  batches.forEach(b => {
    b.files.forEach(f => {
      const { prefix } = getNodeTypeAndPrefix(f.path, f.fileCategory);
      pathPrefixMap[f.path] = prefix;
    });
  });
  
  batches.forEach((batch, idx) => {
    const batchIndex = batch.batchIndex;
    console.log(`\n--- Processing Batch ${idx + 1}/${batches.length} (Index: ${batchIndex}, Files: ${batch.files.length}) ---`);
    
    // Prepare input JSON for extract-structure.mjs
    const analyzerInput = {
      projectRoot: PROJECT_ROOT,
      batchFiles: batch.files,
      batchImportData: batch.batchImportData
    };
    
    const inputPath = path.join(PROJECT_ROOT, `.understand-anything/tmp/ua-file-analyzer-input-${batchIndex}.json`);
    const outputPath = path.join(PROJECT_ROOT, `.understand-anything/tmp/ua-file-extract-results-${batchIndex}.json`);
    
    fs.writeFileSync(inputPath, JSON.stringify(analyzerInput, null, 2));
    
    // Run extract-structure.mjs
    console.log(`  Running extract-structure.mjs...`);
    try {
      execSync(`node ${SKILL_DIR}/extract-structure.mjs "${inputPath}" "${outputPath}"`, {
        env: { ...process.env, PLUGIN_ROOT: '/home/mb10856/.understand-anything-plugin' },
        stdio: 'inherit'
      });
    } catch (err) {
      console.error(`  Error running extract-structure.mjs for batch ${batchIndex}:`, err.message);
      return;
    }
    
    // Read results
    if (!fs.existsSync(outputPath)) {
      console.error(`  Output file missing for batch ${batchIndex}!`);
      return;
    }
    const extractResults = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    
    // Generate nodes and edges
    const nodes = [];
    const edges = [];
    
    extractResults.results.forEach(fileResult => {
      const { path: filePath, fileCategory, totalLines } = fileResult;
      const { type: nodeType, prefix } = getNodeTypeAndPrefix(filePath, fileCategory);
      
      const fileNodeId = `${prefix}${filePath}`;
      const complexity = totalLines < 50 ? 'simple' : (totalLines < 200 ? 'moderate' : 'complex');
      
      // 1. Add file node
      const fileNode = {
        id: fileNodeId,
        type: nodeType,
        name: path.basename(filePath),
        filePath: filePath,
        summary: generateSummary(filePath, fileCategory, fileResult),
        tags: generateTags(filePath, fileCategory, (fileResult.exports || []).length),
        complexity: complexity
      };
      
      if (fileCategory === 'code' && fileResult.language === 'python') {
        fileNode.languageNotes = 'Python script executing loop analysis or model training components.';
      }
      
      nodes.push(fileNode);
      
      // 2. Process functions
      const funcNodesList = fileResult.functions || [];
      funcNodesList.forEach(fn => {
        const fnName = fn.name;
        // Significance filter: lines >= 10 or exported
        const fnLines = fn.endLine - fn.startLine;
        const isExported = (fileResult.exports || []).some(exp => exp.name === fnName);
        if (fnLines >= 10 || isExported) {
          const fnId = `function:${filePath}:${fnName}`;
          const fnComplexity = fnLines < 30 ? 'simple' : (fnLines < 100 ? 'moderate' : 'complex');
          nodes.push({
            id: fnId,
            type: 'function',
            name: fnName,
            filePath: filePath,
            lineRange: [fn.startLine, fn.endLine],
            summary: `Function implementing the execution logic for '${fnName}'.`,
            tags: ['function', 'logic-block', 'helper'],
            complexity: fnComplexity
          });
          
          // Contains edge
          edges.push({
            source: fileNodeId,
            target: fnId,
            type: 'contains',
            direction: 'forward',
            weight: 1.0
          });
          
          if (isExported) {
            edges.push({
              source: fileNodeId,
              target: fnId,
              type: 'exports',
              direction: 'forward',
              weight: 0.8
            });
          }
        }
      });
      
      // 3. Process classes
      const classNodesList = fileResult.classes || [];
      classNodesList.forEach(cls => {
        const clsName = cls.name;
        const clsLines = cls.endLine - cls.startLine;
        const isExported = (fileResult.exports || []).some(exp => exp.name === clsName);
        if (clsLines >= 20 || (cls.methods || []).length >= 2 || isExported) {
          const clsId = `class:${filePath}:${clsName}`;
          const clsComplexity = clsLines < 50 ? 'simple' : (clsLines < 200 ? 'moderate' : 'complex');
          nodes.push({
            id: clsId,
            type: 'class',
            name: clsName,
            filePath: filePath,
            lineRange: [cls.startLine, cls.endLine],
            summary: `Class defining properties and operations for '${clsName}'.`,
            tags: ['class-definition', 'object-oriented', 'structure'],
            complexity: clsComplexity
          });
          
          // Contains edge
          edges.push({
            source: fileNodeId,
            target: clsId,
            type: 'contains',
            direction: 'forward',
            weight: 1.0
          });
          
          if (isExported) {
            edges.push({
              source: fileNodeId,
              target: clsId,
              type: 'exports',
              direction: 'forward',
              weight: 0.8
            });
          }
        }
      });
      
      // 4. Create imports edges (1:1 with batchImportData)
      const imports = batch.batchImportData[filePath] || [];
      imports.forEach(impPath => {
        const targetPrefix = pathPrefixMap[impPath] || 'file:';
        edges.push({
          source: fileNodeId,
          target: `${targetPrefix}${impPath}`,
          type: 'imports',
          direction: 'forward',
          weight: 0.7
        });
      });
      
      // 5. Create some semantic relations based on path conventions
      if (fileCategory === 'config' && filePath.startsWith('config/')) {
        // config files configure the training/evaluation scripts
        edges.push({
          source: fileNodeId,
          target: 'file:rl_autoschedular/rl_autoschedular_paper/train.py',
          type: 'configures',
          direction: 'forward',
          weight: 0.6
        });
      }
      
      if (fileCategory === 'docs' && filePath.startsWith('docs/')) {
        // docs document the codebase
        edges.push({
          source: fileNodeId,
          target: 'document:README.md',
          type: 'documents',
          direction: 'forward',
          weight: 0.5
        });
      }
    });
    
    // Check if we need to split this batch
    const nodeCount = nodes.length;
    const edgeCount = edges.length;
    console.log(`  Generated ${nodeCount} nodes, ${edgeCount} edges`);
    
    if (nodeCount <= 60 && edgeCount <= 120) {
      // Single-part write
      const outBatchPath = path.join(PROJECT_ROOT, `.understand-anything/intermediate/batch-${batchIndex}.json`);
      fs.writeFileSync(outBatchPath, JSON.stringify({ nodes, edges }, null, 2));
      console.log(`  Wrote batch-${batchIndex}.json`);
    } else {
      // Multi-part write
      const parts = Math.ceil(Math.max(nodeCount / 60, edgeCount / 120));
      console.log(`  Splitting batch into ${parts} parts...`);
      
      // Simple sorting of files in batch to partition deterministically
      const uniqueFiles = Array.from(new Set(nodes.filter(n => n.filePath).map(n => n.filePath))).sort();
      const filesPerPart = Math.ceil(uniqueFiles.length / parts);
      
      for (let k = 0; k < parts; k++) {
        const partFiles = uniqueFiles.slice(k * filesPerPart, (k + 1) * filesPerPart);
        const fileSet = new Set(partFiles);
        
        const partNodes = nodes.filter(n => fileSet.has(n.filePath));
        const partNodeIds = new Set(partNodes.map(n => n.id));
        
        const partEdges = edges.filter(e => partNodeIds.has(e.source));
        
        const outPartPath = path.join(PROJECT_ROOT, `.understand-anything/intermediate/batch-${batchIndex}-part-${k + 1}.json`);
        fs.writeFileSync(outPartPath, JSON.stringify({ nodes: partNodes, edges: partEdges }, null, 2));
        console.log(`  Wrote batch-${batchIndex}-part-${k + 1}.json (${partNodes.length} nodes, ${partEdges.length} edges)`);
      }
    }
  });
  
  console.log('\nAll batches processed successfully!');
}

main();
