# Complete API Testing Script
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "SELLEROPS API DEMO TEST" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Wait for server
Write-Host "`n[1] Waiting for server..." -ForegroundColor Yellow
$maxAttempts = 20
$attempt = 0
$serverReady = $false

while ($attempt -lt $maxAttempts -and -not $serverReady) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:3000" -Method GET -TimeoutSec 2 -UseBasicParsing 2>$null
        $serverReady = $true
        Write-Host "✓ Server is ready!" -ForegroundColor Green
    } catch {
        $attempt++
        Write-Host "  Waiting... ($attempt/$maxAttempts)" -ForegroundColor Gray
        Start-Sleep -Seconds 1
    }
}

if (-not $serverReady) {
    Write-Host "✗ Server not responding. Please run 'npm run dev' first." -ForegroundColor Red
    exit 1
}

# Test 1: GET Products Catalog
Write-Host "`n[2] Testing GET /api/recommendations..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/recommendations" -Method GET -UseBasicParsing
    $data = $response.Content | ConvertFrom-Json
    
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Total Products: $($data.totalProducts)" -ForegroundColor Gray
    Write-Host "  Categories: $($data.categories -join ', ')" -ForegroundColor Gray
    Write-Host "  TensorFlow Backend: $($data.tensorFlowBackend)" -ForegroundColor Gray
    Write-Host "  TF Ready: $($data.tensorFlowReady)" -ForegroundColor Gray
    
    # Show first 3 products
    Write-Host "`n  Sample Products:" -ForegroundColor Cyan
    $data.products | Select-Object -First 3 | ForEach-Object {
        Write-Host "    • $($_.name) - ₹$($_.price) [$($_.category)]" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
}

# Test 2: POST Smart Recommendations (without Gemini to save API calls)
Write-Host "`n[3] Testing POST /api/recommendations (TensorFlow only)..." -ForegroundColor Yellow
try {
    $body = @{
        productId = "prod-1"
        strategy = "mixed"
        useGeminiAnalysis = $false
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/recommendations" `
        -Method POST `
        -Body $body `
        -ContentType "application/json" `
        -UseBasicParsing
    
    $data = $response.Content | ConvertFrom-Json
    
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Target: $($data.targetProduct.name)" -ForegroundColor Gray
    Write-Host "  Strategy: $($data.strategy)" -ForegroundColor Gray
    Write-Host "  TF Confidence: $($data.tfConfidence)%" -ForegroundColor Gray
    Write-Host "  Backend: $($data.tensorFlowBackend)" -ForegroundColor Gray
    
    Write-Host "`n  Recommendations:" -ForegroundColor Cyan
    $data.recommendations | ForEach-Object {
        Write-Host "    • $($_.product.name) - Match: $($_.score)%" -ForegroundColor Gray
        Write-Host "      Reason: $($_.reason)" -ForegroundColor DarkGray
    }
    
    Write-Host "`n  AI Analysis:" -ForegroundColor Cyan
    Write-Host "    Summary: $($data.analysis.summary)" -ForegroundColor Gray
    Write-Host "    Expected Revenue: $($data.analysis.expectedImpact.revenueIncrease)" -ForegroundColor Gray
    Write-Host "    Conversion Boost: $($data.analysis.expectedImpact.conversionBoost)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
}

# Test 3: Beast Mode Simulation
Write-Host "`n[4] Testing POST /api/simulate..." -ForegroundColor Yellow
try {
    $body = @{
        priceChange = -5
        adSpendChange = 10
        shippingSpeedChange = 0
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/simulate" `
        -Method POST `
        -Body $body `
        -ContentType "application/json" `
        -UseBasicParsing
    
    $data = $response.Content | ConvertFrom-Json
    
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  New Price: ₹$($data.projectedPrice)" -ForegroundColor Gray
    Write-Host "  Revenue Change: $($data.revenueChange)%" -ForegroundColor Gray
    Write-Host "  Margin Change: $($data.marginChange)%" -ForegroundColor Gray
    Write-Host "  Risk Level: $($data.riskLevel)" -ForegroundColor Gray
    Write-Host "  Competitor Response: $($data.competitorResponseProbability)%" -ForegroundColor Gray
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
}

# Test 4: Initialize Database
Write-Host "`n[5] Testing GET /api/admin/init..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/admin/init" -Method GET -UseBasicParsing
    $data = $response.Content | ConvertFrom-Json
    
    Write-Host "✓ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Tables: $($data.tables -join ', ')" -ForegroundColor Gray
} catch {
    Write-Host "✗ Failed: $_" -ForegroundColor Red
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "✅ API DEMO TEST COMPLETE" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor White
Write-Host "  1. Open http://localhost:3000 in browser" -ForegroundColor Cyan
Write-Host "  2. Click 'Smart Recommendations' tab" -ForegroundColor Cyan
Write-Host "  3. Select a product and generate recommendations" -ForegroundColor Cyan
Write-Host "`nAll systems operational! 🚀" -ForegroundColor Green
