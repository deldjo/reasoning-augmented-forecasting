import sys
import argparse
import json
import numpy as np
import pandas as pd
import tomllib
import os
import requests
import warnings
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="My Agent (ARIMA + Text) for Track 2")
    parser.add_argument("--panels", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_text", action="store_true", help="开启文本推理模式")
    return parser.parse_args()

def find_file(directory, filename):
    if os.path.exists(os.path.join(directory, filename)):
        return os.path.join(directory, filename)
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_parquet_files(directory):
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet") and "golden" not in root:
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def analyze_text(text_dir):
    """调用大模型API提取文本信号（含关键词回退逻辑）"""
    text_files = []
    if os.path.exists(text_dir):
        for root, dirs, files in os.walk(text_dir):
            for file in files:
                if file.endswith(".txt") or file.endswith(".json"):
                    text_files.append(os.path.join(root, file))
    
    all_text = ""
    for tf in text_files:
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                all_text += f.read() + "\n"
        except:
            pass
    
    all_text = all_text[:3000]
    
    # 默认初始化（确保此处必定定义）
    sentiment_score = 0.0
    risk_flag = False

    endpoint = os.environ.get("MODEL_ENDPOINT")
    model_name = os.environ.get("MODEL_NAME", "qwen2.5-72b-instruct")
    
    # 如果没有 API，使用关键词逻辑
    if not endpoint:
        hawkish_words = ["rate hike", "tightening", "inflation risk", "hawkish", "surplus"]
        dovish_words = ["rate cut", "easing", "recession", "dovish", "deficit"]
        shock_words = ["unexpected", "crisis", "plunge", "surge", "emergency"]
        for w in hawkish_words:
            if w in all_text.lower(): sentiment_score += 0.2
        for w in dovish_words:
            if w in all_text.lower(): sentiment_score -= 0.2
        for w in shock_words:
            if w in all_text.lower(): risk_flag = True
        return sentiment_score, risk_flag

    # 正式调用 API
    try:
        prompt = f"""
请根据以下金融新闻/政策文本，判断其对宏观市场的整体情绪：
1. 输出一个介于 -1 到 1 之间的“情绪分数”（-1为极度利空，1为极度利多）。
2. 如果存在极端风险事件，输出“true”，否则输出“false”。
仅输出一个 JSON 格式：{{"sentiment": -0.6, "risk": true}}

文本内容：
{all_text}
"""
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = content.strip().strip("```json").strip("```")
            data = json.loads(content)
            return float(data["sentiment"]), bool(data["risk"])
        else:
            print(f"API 返回错误代码: {response.status_code}")
            return 0.0, False
    except Exception as e:
        print(f"API 调用失败，已回退到本地逻辑: {e}")
        # 回退逻辑
        hawkish_words = ["rate hike", "tightening", "inflation risk", "hawkish", "surplus"]
        dovish_words = ["rate cut", "easing", "recession", "dovish", "deficit"]
        shock_words = ["unexpected", "crisis", "plunge", "surge", "emergency"]
        for w in hawkish_words:
            if w in all_text.lower(): sentiment_score += 0.2
        for w in dovish_words:
            if w in all_text.lower(): sentiment_score -= 0.2
        for w in shock_words:
            if w in all_text.lower(): risk_flag = True
        return sentiment_score, risk_flag

def main():
    # 兼容 Docker 传入的 forecast 命令
    if len(sys.argv) > 1 and sys.argv[1] == "forecast":
        sys.argv.pop(1)

    args = parse_args()
    
    card_path = find_file(args.panels, "card.toml")
    if card_path is None:
        parent_dir = os.path.dirname(args.panels.rstrip('/'))
        card_path = find_file(parent_dir, "card.toml")
    if card_path is None:
        raise FileNotFoundError("找不到 card.toml")
    
    with open(card_path, "rb") as f:
        card = tomllib.load(f)
    
    unit_id = card["task"]["id"]
    asof = args.asof
    asset_ids = card["targets"]["asset_ids"]
    horizons = card["targets"]["horizons"]
    n_draws = 500
    
    parquet_files = find_parquet_files(args.panels)
    all_panels = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= pd.to_datetime(asof)]
        all_panels.append(df)
    
    data = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()
    
    sentiment_score = 0.0
    risk_flag = False
    if args.use_text:
        sentiment_score, risk_flag = analyze_text(args.text)
        print(f"[文本推理模式] 情绪分数: {sentiment_score:.2f}, 尾部风险: {risk_flag}")
    else:
        print("[纯数值模式] 执行ARIMA基线预测")
    
    forecast_rows = []
    for asset in asset_ids:
        asset_data = data[data["asset"] == asset].sort_values("date")
        try:
            if len(asset_data) < 20:
                raise ValueError("样本太少")
            model = ARIMA(asset_data["value"], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=max(horizons))
            mean_forecast = forecast.predicted_mean
            std_forecast = forecast.se_mean
            
            for horizon in horizons:
                mean_val = float(mean_forecast.iloc[horizon-1])
                std_val = float(std_forecast.iloc[horizon-1])
                if pd.isna(std_val) or std_val == 0:
                    std_val = 0.01
                if args.use_text:
                    mean_val = mean_val * (1 + sentiment_score * 0.05)
                    if risk_flag:
                        std_val = std_val * 1.5
                samples = np.random.normal(mean_val, std_val, n_draws)
                for d in range(n_draws):
                    forecast_rows.append({
                        "draw": d, "asset": asset, "horizon": horizon, "value": float(samples[d])
                    })
        except Exception as e:
            # ARIMA失败，回退到随机游走
            mean_val = asset_data["value"].iloc[-1] if len(asset_data) > 0 else 0.0
            std_val = asset_data["value"].tail(20).std() if len(asset_data) > 10 else 0.01
            if pd.isna(std_val) or std_val == 0:
                std_val = 0.01
            for horizon in horizons:
                samples = np.random.normal(mean_val, std_val, n_draws)
                for d in range(n_draws):
                    forecast_rows.append({"draw": d, "asset": asset, "horizon": horizon, "value": float(samples[d])})
    
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    forecast_path = os.path.join(out_dir, "forecast.parquet")
    meta_path = os.path.join(out_dir, "forecast_meta.json")
    rationale_path = os.path.join(out_dir, "forecast_rationale.md")
    
    pd.DataFrame(forecast_rows).to_parquet(forecast_path, index=False)
    
    meta = {"unit_id": unit_id, "asof": asof, "asset_ids": asset_ids, "horizons": horizons, "representation": "samples", "n_draws": n_draws}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    with open(rationale_path, "w") as f:
        f.write(f"# {unit_id}\n\nAS OF Date: {asof}\n\n## 推理逻辑\n")
        f.write(f"当前模型: {'ARIMA+LLM文本推理' if args.use_text else 'ARIMA基线'}\n")
        f.write(f"文本情绪分数: {sentiment_score}, 尾部风险: {risk_flag}\n")

if __name__ == "__main__":
    main()