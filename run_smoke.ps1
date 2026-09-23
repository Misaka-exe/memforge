# MemForge Stage 5B - 10-question real Smoke (DeepSeek)
# Window stays open; user can watch live progress.

Set-Location D:\agentmemory\memforge
$env:HF_HOME = "D:\agentmemory\memforge\.hf_cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:MEMFORGE_LLM_MODEL = "deepseek-chat"
$env:MEMFORGE_LLM_BASE_URL = "https://api.deepseek.com/v1"

Write-Host ""
Write-Host "=== MemForge Stage 5B: 10-question Real Smoke (DeepSeek) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment ready:"
Write-Host "  Model:    deepseek-chat"
Write-Host "  Endpoint: https://api.deepseek.com/v1"
Write-Host "  Data:     LongMemEval-S (500 questions, 10 for smoke)"
Write-Host ""

$key = Read-Host "Paste your DeepSeek API key (sk-...)"
$env:MEMFORGE_LLM_API_KEY = $key.Trim()
Write-Host ""
Write-Host "API key set (this window only, not saved to any file)." -ForegroundColor Green
Write-Host "Running 10-question smoke test..." -ForegroundColor Green
Write-Host ""

Write-Host "Output dir: results\longmemeval\stage5_real_smoke (separate from Mock results)"
Write-Host ""

.venv\Scripts\python.exe -m benchmarks.scripts.run_longmemeval_qa --smoke --provider openai --model all-MiniLM-L6-v2 --out results\longmemeval\stage5_real_smoke

Write-Host ""
Write-Host "=== Smoke run finished ===" -ForegroundColor Cyan
Write-Host "Result file: results\longmemeval\stage5_real_smoke\naive_vector\raw_qa.jsonl"
Write-Host ""
Write-Host "Send the final output (metrics summary) back to the AI for the 5-point check."
Write-Host "Do NOT close this window yet."
Write-Host ""
Read-Host "Press Enter to close this window"
