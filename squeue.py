import subprocess

def run_command():
    command = 'squeue | grep $USER'
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout.strip('\n'))
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)

if __name__ == "__main__":
    run_command()
