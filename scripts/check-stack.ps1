# SellerOps Tech Stack Verification
Write-Host "=== SELLEROPS TECH STACK CHECK ===" -ForegroundColor Cyan

# Node
Write-Host "`n[1] Node.js:" -ForegroundColor Yellow
node --version

# Dependencies
Write-Host "`n[2] Key Dependencies:" -ForegroundColor Yellow
npm list @tensorflow/tfjs --depth=0 2>$null | Select-String "tfjs"
npm list @google/genai --depth=0 2>$null | Select-String "genai"
npm list ioredis --depth=0 2>$null | Select-String "ioredis"

# Docker Redis
Write-Host "`n[3] Redis Docker:" -ForegroundColor Yellow
docker ps --filter "name=seller-ops-redis" --format "Status: {{.Status}}"

# Environment
Write-Host "`n[4] Environment:" -ForegroundColor Yellow
if (Test-Path ".env") {
    $env = Get-Content ".env" | Select-String "GEMINI_API_KEY"
    if ($env) { Write-Host "GEMINI_API_KEY: Configured" -ForegroundColor Green }
}

# Files
Write-Host "`n[5] Implementation Files:" -ForegroundColor Yellow
$files = @(
    "src/lib/tensorflow/recommendation-engine.ts",
    "src/lib/gemini/recommendation-analysis.ts",
    "src/app/api/recommendations/route.ts"
)
foreach ($f in $files) {
    if (Test-Path $f) { Write-Host "  OK: $f" -ForegroundColor Green }
}

# Tests
Write-Host "`n[6] Running Tests..." -ForegroundColor Yellow
npm test -- --run 2>&1 | Select-String "(passed|failed)" | Select-Object -First 3

Write-Host "`n=== READY TO START ===" -ForegroundColor Cyan
Write-Host "Run: npm run dev" -ForegroundColor Green
