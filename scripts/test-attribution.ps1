# Quick AI Attribution Test Script
# Tests if Gemini AI analysis is working

Write-Host "`n╔═══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  SellerOps AI Attribution Test                      ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Test data
$testPayload = @"
{
  "threat": {
    "id": "test-threat-001",
    "title": "Critical Price Drop - Competitor Action",
    "type": "COMPETITOR_PRICE_DROP",
    "severity": "CRITICAL",
    "description": "TechGiant Corp reduced price by 18.5% on Wireless Earbuds Pro",
    "detectedAt": "$(Get-Date -Format 'o')"
  },
  "signals": [
    {
      "id": "signal-001",
      "type": "COMPETITOR_PRICE",
      "timestamp": "$((Get-Date).AddHours(-1).ToString('o'))",
      "value": 2036,
      "previousValue": 2499,
      "delta": -463
    },
    {
      "id": "signal-002",
      "type": "CONVERSION_DROP",
      "timestamp": "$((Get-Date).AddMinutes(-30).ToString('o'))",
      "value": 2.1,
      "previousValue": 3.8,
      "delta": -1.7
    }
  ]
}
"@

Write-Host "Sending threat to Gemini AI for analysis..." -ForegroundColor Yellow
Write-Host "⏳ This may take 10-15 seconds...`n" -ForegroundColor Gray

try {
    $response = Invoke-RestMethod `
        -Uri "http://localhost:3000/api/attribution" `
        -Method POST `
        -Body $testPayload `
        -ContentType "application/json" `
        -TimeoutSec 30
    
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║  ✓✓✓ AI ATTRIBUTION ANALYSIS SUCCESSFUL! ✓✓✓        ║" -ForegroundColor Green  
    Write-Host "╚═══════════════════════════════════════════════════════╝`n" -ForegroundColor Green
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "CONFIDENCE SCORE: $($response.confidence)%" -ForegroundColor Yellow -BackgroundColor DarkBlue
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan
    
    Write-Host "📊 EXECUTIVE SUMMARY:" -ForegroundColor Cyan
    Write-Host "  $($response.summary)`n" -ForegroundColor White
    
    Write-Host "🔍 ROOT CAUSES IDENTIFIED:" -ForegroundColor Red
    foreach ($cause in $response.causes) {
        Write-Host "  ► [$($cause.impact)] $($cause.factor)" -ForegroundColor White
        Write-Host "    Evidence: $($cause.evidence)`n" -ForegroundColor Gray
    }
    
    Write-Host "💡 STRATEGIC RECOMMENDATIONS:" -ForegroundColor Green
    foreach ($action in $response.suggestedActions) {
        Write-Host "  ► [$($action.priority)] $($action.action)" -ForegroundColor White
        Write-Host "    Expected Outcome: $($action.expectedOutcome)`n" -ForegroundColor Gray
    }
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "✓ Gemini AI is working correctly!" -ForegroundColor Green
    Write-Host "✓ Attribution analysis is operational" -ForegroundColor Green
    Write-Host "✓ Ready for demo presentation`n" -ForegroundColor Green
    
} catch {
    Write-Host "╔═══════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║  ✗✗✗ AI ATTRIBUTION FAILED ✗✗✗                      ║" -ForegroundColor Red
    Write-Host "╚═══════════════════════════════════════════════════════╝`n" -ForegroundColor Red
    
    $errorMsg = $_.Exception.Message
    Write-Host "Error: $errorMsg`n" -ForegroundColor Red
    
    if ($errorMsg -match "API key|403|401") {
        Write-Host "🔧 TROUBLESHOOTING:" -ForegroundColor Yellow
        Write-Host "  1. Check if GEMINI_API_KEY is in .env.local" -ForegroundColor Cyan
        Write-Host "  2. Verify the API key is valid" -ForegroundColor Cyan
        Write-Host "  3. Restart the dev server: npm run dev`n" -ForegroundColor Cyan
    } elseif ($errorMsg -match "503|overloaded") {
        Write-Host "⚠️  Gemini API is temporarily overloaded. Try again in a moment.`n" -ForegroundColor Yellow
    } else {
        Write-Host "Check server logs for details.`n" -ForegroundColor Yellow
    }
    
    exit 1
}
