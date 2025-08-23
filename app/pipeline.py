"""
Replace `run_pipeline` with your real PPT/PDF -> (.apkg, .csv) generator.
Return absolute file paths to the outputs you created.
Keep it pure; main.py handles HTTP + temp files.
"""
from pathlib import Path

def run_pipeline(input_path: str) -> tuple[str, str]:
    input_file = Path(input_path)
    out_apkg = input_file.with_suffix(input_file.suffix + ".apkg")
    out_csv  = input_file.with_suffix(input_file.suffix + ".csv")

    # DEMO OUTPUTS so the API works now; replace with your real logic.
    out_apkg.write_bytes(b"fake .apkg content")
    out_csv.write_text("question,answer\nExample?,Yes\n", encoding="utf-8")

    return str(out_apkg), str(out_csv)
