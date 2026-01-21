# Simple API Test Script
Write-Host "=== SELLEROPS API TEST ===" -ForegroundColor Cyan

# Wait for server
Write-Host "`nWaiting for server..." -ForegroundColor Yellow
$maxAttempts = 15
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    try {
        $test = Invoke-WebRequest -Uri "http://localhost:3000" -Method GET -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        $ready = $true
        Write-Host "Server ready!" -ForegroundColor Green
    } catch {
        $attempt++
        Write-Host "Attempt $attempt..." -ForegroundColor Gray
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "Server not responding." -ForegroundColor Red
    exit 1
}

# Test 1: Get Products
Write-Host "`n[1] GET /api/recommendations" -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:3000/api/recommendations" -Method GET -UseBasicParsing
    $d = $r.Content | ConvertFrom-Json
    Write-Host "OK - $($d.totalProducts) products, TF: $($d.tensorFlowBackend)" -ForegroundColor Green
} catch {
    Write-Host "FAIL: $_" -ForegroundColor Red
}

# Test 2: TensorFlow Recommendations
Write-Host "`n[2] POST /api/recommendations (TensorFlow)" -ForegroundColor Yellow
try {
    $body = '{"productId":"prod-1","strategy":"mixed","useGeminiAnalysis":false}'
    $r = Invoke-WebRequest -Uri "http://localhost:3000/api/recommendations" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
    $d = $r.Content | ConvertFrom-Json
    Write-Host "OK - $($d.recommendations.Count) recommendations" -ForegroundColor Green
    Write-Host "Target: $($d.targetProduct.name)" -ForegroundColor Gray
    $d.recommendations | ForEach-Object { Write-Host "  • $($_.product.name) - $($_.score)%" -ForegroundColor Gray }
} catch {
    Write-Host "FAIL: $_" -ForegroundColor Red
}

# Test 3: Simulation
Write-Host "`n[3] POST /api/simulate" -ForegroundColor Yellow
try {
    $body = '{"priceChange":-5,"adSpendChange":10,"shippingSpeedChange":0}'
    $r = Invoke-WebRequest -Uri "http://localhost:3000/api/simulate" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
    $d = $r.Content | ConvertFrom-Json
    Write-Host "OK - Risk: $($d.riskLevel), Revenue: $($d.revenueChange)%" -ForegroundColor Green
} catch {
    Write-Host "FAIL: $_" -ForegroundColor Red
}

# Test 4: DB Init
Write-Host "`n[4] GET /api/admin/init" -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:3000/api/admin/init" -Method GET -UseBasicParsing
    Write-Host "OK - Database initialized" -ForegroundColor Green
} catch {
    Write-Host "FAIL: $_" -ForegroundColor Red
}

Write-Host "`n=== ALL TESTS COMPLETE ===" -ForegroundColor Cyan
Write-Host "Open: http://localhost:3000" -ForegroundColor Green
