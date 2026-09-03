import os
import subprocess
import pandas as pd
import re

units_dir = "units"
unit_dirs = [d for d in os.listdir(units_dir) if os.path.isdir(os.path.join(units_dir, d))]

results = []

for unit in unit_dirs:
    unit_path = os.path.join(units_dir, unit)
    card_path = os.path.join(unit_path, "card.toml")
    
    if not os.path.exists(card_path):
        continue

    try:
        with open(card_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 用正则抓取日期格式为 YYYY-MM-DD 的值，作为 asof
        match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
        if match:
            asof = match.group(1)
        else:
            print(f"Skipping {unit}: asof not found")
            continue
    except Exception as e:
        print(f"Skipping {unit}: {e}")
        continue
    
    out_dir = os.path.join("out_batch", unit)
    os.makedirs(out_dir, exist_ok=True)
    
    for use_text in [False, True]:
        mode = "text" if use_text else "blind"
        command = [
            "python", "my_agent.py",
            "--panels", unit_path,
            "--text", os.path.join(unit_path, "text"),
            "--asof", asof,
            "--out", os.path.join(out_dir, f"forecast_{mode}.parquet")
        ]
        if use_text:
            command.append("--use_text")
        
        print(f"Running {unit} [{mode}] with asof={asof}...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"  -> Success!")
            results.append({"unit": unit, "asof": asof, "mode": mode, "status": "pass"})
        else:
            print(f"  -> Error: {result.stderr[-300:]}")
            results.append({"unit": unit, "asof": asof, "mode": mode, "status": "fail"})

# 保存结果记录
df = pd.DataFrame(results)
df.to_csv("ablation_results.csv", index=False)
print("批量运行完成！结果已保存至 ablation_results.csv")
