import re

with open("docs/VERSIONS.md", "r") as f:
    text = f.read()

text = text.replace("redirected to `rl_autoschedular_v4`", "redirected to `rl_autoschedular_v5`")

v4_start = text.find("### V4 - Combined")
v4_5_start = text.find("### V4.5 - Robust Integration")
v5_start = text.find("### V5 - Action Space Expansion")

if v5_start < v4_start:
    # V5 is before V4
    # Let's extract V5 block
    v5_block = text[v5_start:v4_start]
    
    # Remove V5 block from current position
    text = text[:v5_start] + text[v4_start:]
    
    # Append V5 block at the end
    text = text + "\n\n" + v5_block

with open("docs/VERSIONS.md", "w") as f:
    f.write(text)
