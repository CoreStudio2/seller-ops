# Demo Data Seeding Script
# Generates realistic signals and threats for the SellerOps War Room

Write-Host "=== SellerOps Demo Data Generator ===" -ForegroundColor Cyan
Write-Host ""

$apiBase = "http://localhost:3000/api"

# Check if server is running
Write-Host "Checking if server is running..." -ForegroundColor Yellow
try {
    $status = Invoke-RestMethod -Uri "$apiBase/status" -Method GET -ErrorAction Stop
    Write-Host "✓ Server is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Server is not running. Please run: npm run dev" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Generating demo signals..." -ForegroundColor Cyan
Write-Host ""

# Function to ingest signal
function Send-Signal {
    param(
        [string]$Type,
        [double]$Value,
        [hashtable]$Meta
    )
    
    $body = @{
        type = $Type
        value = $Value
        meta = $Meta
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "$apiBase/ingest" -Method POST -Body $body -ContentType "application/json"
        Write-Host "  ✓ " -NoNewline -ForegroundColor Green
        Write-Host "$Type " -NoNewline -ForegroundColor White
        if ($response.threat) {
            Write-Host "[THREAT DETECTED: $($response.threat.severity)]" -ForegroundColor Red
        } else {
            Write-Host "[Normal]" -ForegroundColor Gray
        }
        Start-Sleep -Milliseconds 500
        return $response
    } catch {
        Write-Host "  ✗ Failed to send $Type" -ForegroundColor Red
        return $null
    }
}

# 1. Competitor Price Drops (Creates Threats)
Write-Host "1. Competitor Activity Signals" -ForegroundColor Yellow
Send-Signal -Type "COMPETITOR_PRICE_DROP" -Value -18.5 -Meta @{
    competitor = "TechGiant Corp"
    product = "Wireless Earbuds Pro"
    oldPrice = 2499
    newPrice = 2036
}

Send-Signal -Type "COMPETITOR_PRICE_DROP" -Value -12.0 -Meta @{
    competitor = "MegaMart"
    product = "USB-C Charger"
    oldPrice = 899
    newPrice = 791
}

Send-Signal -Type "COMPETITOR_PRICE_DROP" -Value -25.0 -Meta @{
    competitor = "BudgetElectronics"
    product = "Bluetooth Speaker"
    oldPrice = 1899
    newPrice = 1424
}

# 2. Inventory Issues (Creates Threats)
Write-Host ""
Write-Host "2. Inventory Management Signals" -ForegroundColor Yellow
Send-Signal -Type "LOW_INVENTORY" -Value 8 -Meta @{
    product = "Portable Power Bank"
    currentStock = 8
    reorderPoint = 50
    warehouse = "DC-EAST"
}

Send-Signal -Type "LOW_INVENTORY" -Value 3 -Meta @{
    product = "Screen Protector"
    currentStock = 3
    reorderPoint = 100
    warehouse = "DC-WEST"
}

# 3. Cart Abandonment (Creates Threats)
Write-Host ""
Write-Host "3. Conversion Optimization Signals" -ForegroundColor Yellow
Send-Signal -Type "HIGH_CART_ABANDONMENT" -Value 78.5 -Meta @{
    sessionCount = 450
    abandonedCarts = 353
    averageCartValue = 3200
    topAbandonedProduct = "Wireless Mouse"
}

Send-Signal -Type "CONVERSION_DROP" -Value -22.0 -Meta @{
    currentRate = 2.8
    previousRate = 3.6
    category = "Electronics"
    affectedProducts = 12
}

# 4. Advertising Costs
Write-Host ""
Write-Host "4. Marketing & Advertising Signals" -ForegroundColor Yellow
Send-Signal -Type "AD_COST_SPIKE" -Value 45.0 -Meta @{
    platform = "Google Ads"
    campaign = "Holiday Electronics Sale"
    oldCPC = 12.50
    newCPC = 18.13
    budget = 50000
}

Send-Signal -Type "AD_COST_SPIKE" -Value 30.0 -Meta @{
    platform = "Meta Ads"
    campaign = "Wireless Accessories"
    oldCPC = 8.20
    newCPC = 10.66
    budget = 30000
}

# 5. Shipping Delays
Write-Host ""
Write-Host "5. Logistics & Fulfillment Signals" -ForegroundColor Yellow
Send-Signal -Type "SHIPPING_DELAY" -Value 3.5 -Meta @{
    carrier = "FastShip Express"
    affectedOrders = 87
    averageDelay = 3.5
    region = "Northeast"
}

# 6. Positive Signals (Revenue Growth)
Write-Host ""
Write-Host "6. Revenue & Growth Signals" -ForegroundColor Yellow
Send-Signal -Type "REVENUE_SPIKE" -Value 28.0 -Meta @{
    category = "Accessories"
    amount = 125000
    previousAmount = 97656
    topProduct = "Phone Case Bundle"
}

Send-Signal -Type "PRODUCT_TRENDING" -Value 150.0 -Meta @{
    product = "Wireless Charging Pad"
    viewIncrease = 150.0
    salesIncrease = 85.0
    socialMentions = 340
}

# 7. Customer Behavior
Write-Host ""
Write-Host "7. Customer Experience Signals" -ForegroundColor Yellow
Send-Signal -Type "HIGH_RETURN_RATE" -Value 15.5 -Meta @{
    product = "Budget Earbuds"
    returnRate = 15.5
    threshold = 10.0
    topReason = "Poor sound quality"
}

# 8. More Realistic Activity
Write-Host ""
Write-Host "8. Additional Market Activity" -ForegroundColor Yellow
Send-Signal -Type "COMPETITOR_PRICE_DROP" -Value -8.0 -Meta @{
    competitor = "OnlineElectronics"
    product = "HDMI Cable"
    oldPrice = 249
    newPrice = 229
}

Send-Signal -Type "PRICE_OPTIMIZATION" -Value 5.5 -Meta @{
    product = "Laptop Sleeve"
    oldPrice = 599
    newPrice = 632
    expectedRevenueIncrease = 12.0
}

Send-Signal -Type "SEASONAL_DEMAND" -Value 35.0 -Meta @{
    season = "Holiday Shopping"
    category = "Electronics Gifts"
    demandIncrease = 35.0
    stockLevel = "Adequate"
}

# Summary
Write-Host ""
Write-Host "=== Demo Data Generation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  • Generated 15+ realistic signals" -ForegroundColor White
Write-Host "  • Multiple threat types created" -ForegroundColor White
Write-Host "  • Revenue and competitor data included" -ForegroundColor White
Write-Host "  • Dashboard should now show active threats" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Visit http://localhost:3000" -ForegroundColor Cyan
Write-Host "  2. Check the Threat Feed (left panel)" -ForegroundColor Cyan
Write-Host "  3. View Live Status Bar (top)" -ForegroundColor Cyan
Write-Host "  4. Click on threats for AI attribution analysis" -ForegroundColor Cyan
Write-Host ""
Write-Host "To re-run this script: .\scripts\generate-demo-data.ps1" -ForegroundColor Gray
