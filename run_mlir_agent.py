import asyncio
from datetime import datetime

from llm_action.src.utils.processing import preprocess_code
from llm_action.src.agent import AgentWrapper

async def test_agent():
    """Test the MLIR LLM Agent with a matmul benchmark."""
    
    # Load the MLIR code
    with open("llm_action/data/matmul/matmul_128_256_128.mlir", "r") as f:
        mlir_code = f.read()
        mlir_code = preprocess_code(mlir_code)
    
    print("=" * 80)
    print("MLIR LLM AGENT TEST")
    print("=" * 80)
    print(f"\n📄 Input Code:")
    print("-" * 80)
    print(mlir_code)
    print("-" * 80)
    print("\n🤖 Agent Starting...\n")
    print("=" * 80)
    
    # Initialize agent
    agent = AgentWrapper()
    
    # Track metrics
    transformations = []
    executions = []
    agent_text = []
    
    # Run the agent and stream output
    async for event in agent.run_stream(mlir_code):
        if event is None:
            continue
            
        event_type = event.get('type')
        
        if event_type == 'run':
            status = event.get('status')
            if status == 'started':
                print("\n🚀 RUN STARTED\n")
            elif status == 'completed':
                print("\n✅ RUN COMPLETED\n")
            elif status == 'error':
                print(f"\n❌ ERROR: {event.get('error')}\n")
        
        elif event_type == 'content':
            content = event.get('content', '')
            agent_text.append(content)
            print(content, end='', flush=True)
        
        elif event_type == 'tool':
            status = event.get('status')
            tool_name = event.get('name')
            
            if status == 'started':
                print(f"\n\n🔧 TOOL CALL: {tool_name}")
                print("-" * 80)
                
                args = event.get('arguments', {})
                if tool_name == 'transform_code':
                    print("📝 Transformation Code:")
                    print(args.get('transformation_code', ''))
                elif tool_name == 'execute_code':
                    code = args.get('code', '')
                    print(f"⚙️  Executing code ({len(code)} chars)")
                print("-" * 80)
            
            elif status == 'completed':
                result = event.get('result')
                print(f"\n✓ Tool Result: {result}")
                
                if tool_name == 'transform_code':
                    transformations.append({
                        'result': result
                    })
                elif tool_name == 'execute_code':
                    if isinstance(result, (list, tuple)) and len(result) == 2:
                        exec_time, success = result
                        executions.append({
                            'time_ns': exec_time,
                            'success': success
                        })
                        print(f"   ⏱️  Execution Time: {exec_time:,} ns")
                        print(f"   {'✅' if success else '❌'} Correctness: {success}")
                print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"\n🔧 Total Transformations: {len(transformations)}")
    print(f"⚙️  Total Executions: {len(executions)}")
    
    if executions:
        print(f"\n⏱️  Execution Times:")
        for i, exec_data in enumerate(executions, 1):
            time_ns = exec_data['time_ns']
            success = exec_data['success']
            status_icon = '✅' if success else '❌'
            print(f"   {i}. {time_ns:,} ns {status_icon}")
        
        # Calculate speedup if we have baseline
        if len(executions) > 1:
            baseline = executions[0]['time_ns']
            best = min(e['time_ns'] for e in executions if e['success'])
            speedup = baseline / best if best > 0 else 0
            print(f"\n🚀 Best Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 80)
    print("Full agent response saved to output.")
    print("=" * 80)
    
    # Optionally save full output
    date_format = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"llm_action/log/test/test_{date_format}.txt", "w") as f:
        f.write("".join(agent_text))
    print(f"\n💾 Full output saved to: llm_action/log/test/test_{date_format}.txt\n")

if __name__ == "__main__":
    asyncio.run(test_agent())