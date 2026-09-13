#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster-Whisper 訪視錄音自動轉逐字稿與 SRT 字幕腳本
使用方式：
    python transcribe_audio.py --audio "path/to/audio.wav" --output "path/to/output_dir"
"""

import argparse
import os
import sys
import time

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("請先安裝 faster-whisper：pip install faster-whisper")
    sys.exit(1)


def format_timestamp(seconds: float) -> str:
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def transcribe_audio(audio_path: str, output_dir: str, model_size: str = "small"):
    if not os.path.exists(audio_path):
        print(f"錯誤：找不到音訊檔案 {audio_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}_逐字稿.txt")
    srt_path = os.path.join(output_dir, f"{base_name}_逐字稿.srt")

    print(f"載入 Faster-Whisper 模型 ({model_size}, int8 量化)...")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"模型載入完成（耗時 {time.time() - t0:.2f} 秒）")

    print(f"開始轉譯音訊：{audio_path}...")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="zh",
        initial_prompt="這是台灣的居家醫療、長照與安寧訪視對話，包含台語、原住民族語及國語日常醫護用語、用藥、生命徵象與生活照護。",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"偵測語言：{info.language}（信心度：{info.language_probability:.2f}）")

    all_text = []
    srt_entries = []
    idx = 1

    for segment in segments:
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        text = segment.text.strip()

        line = f"[{start_str[:8]}] {text}"
        all_text.append(line)

        srt_entry = f"{idx}\n{start_str} --> {end_str}\n{text}\n"
        srt_entries.append(srt_entry)

        if idx % 20 == 0:
            print(f"[{start_str[:8]}] {text}")
        idx += 1

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    total_time = time.time() - t0
    print(f"\n轉譯完成！共處理 {idx-1} 句（總耗時：{total_time:.2f} 秒）")
    print(f"已儲存為：\n  - {txt_path}\n  - {srt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訪視錄音自動轉逐字稿工具")
    parser.add_argument("--audio", required=True, help="音訊檔案路徑 (.wav/.mp3/.m4a)")
    parser.add_argument("--output", default=".", help="輸出目錄路徑")
    parser.add_argument("--model", default="small", help="模型大小 (tiny/base/small/medium)")
    args = parser.parse_args()

    transcribe_audio(args.audio, args.output, args.model)
