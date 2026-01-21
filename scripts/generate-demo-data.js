#!/usr/bin/env node

/**
 * Demo Data Generator - Node.js Version
 * Generates realistic signals and threats for SellerOps War Room
 * Usage: node scripts/generate-demo-data.js
 */

const API_BASE = 'http://localhost:3000/api';

// Color output helpers
const colors = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  gray: '\x1b[90m',
  white: '\x1b[37m'
};

function log(message, color = 'white') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

// Sleep helper
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Function to send signal
async function sendSignal(type, value, meta) {
  const body = JSON.stringify({ type, value, meta });
  
  try {
    const response = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    
    const data = await response.json();
    
    process.stdout.write(colors.green + '  ✓ ' + colors.reset);
    process.stdout.write(colors.white + type + ' ' + colors.reset);
    
    if (data.threat) {
      log(`[THREAT DETECTED: ${data.threat.severity}]`, 'red');
    } else {
      log('[Normal]', 'gray');
    }
    
    await sleep(500);
    return data;
  } catch (error) {
    log(`  ✗ Failed to send ${type}: ${error.message}`, 'red');
    return null;
  }
}

// Main execution
async function main() {
  log('\n=== SellerOps Demo Data Generator ===\n', 'cyan');
  
  // Check if server is running
  log('Checking if server is running...', 'yellow');
  try {
    const response = await fetch(`${API_BASE}/status`);
    if (response.ok) {
      log('✓ Server is running\n', 'green');
    } else {
      throw new Error('Server returned error');
    }
  } catch (error) {
    log('✗ Server is not running. Please run: npm run dev\n', 'red');
    process.exit(1);
  }
  
  log('Generating demo signals...\n', 'cyan');
  
  // 1. Competitor Price Drops
  log('1. Competitor Activity Signals', 'yellow');
  await sendSignal('COMPETITOR_PRICE_DROP', -18.5, {
    competitor: 'TechGiant Corp',
    product: 'Wireless Earbuds Pro',
    oldPrice: 2499,
    newPrice: 2036
  });
  
  await sendSignal('COMPETITOR_PRICE_DROP', -12.0, {
    competitor: 'MegaMart',
    product: 'USB-C Charger',
    oldPrice: 899,
    newPrice: 791
  });
  
  await sendSignal('COMPETITOR_PRICE_DROP', -25.0, {
    competitor: 'BudgetElectronics',
    product: 'Bluetooth Speaker',
    oldPrice: 1899,
    newPrice: 1424
  });
  
  // 2. Inventory Issues
  log('\n2. Inventory Management Signals', 'yellow');
  await sendSignal('LOW_INVENTORY', 8, {
    product: 'Portable Power Bank',
    currentStock: 8,
    reorderPoint: 50,
    warehouse: 'DC-EAST'
  });
  
  await sendSignal('LOW_INVENTORY', 3, {
    product: 'Screen Protector',
    currentStock: 3,
    reorderPoint: 100,
    warehouse: 'DC-WEST'
  });
  
  // 3. Cart Abandonment
  log('\n3. Conversion Optimization Signals', 'yellow');
  await sendSignal('HIGH_CART_ABANDONMENT', 78.5, {
    sessionCount: 450,
    abandonedCarts: 353,
    averageCartValue: 3200,
    topAbandonedProduct: 'Wireless Mouse'
  });
  
  await sendSignal('CONVERSION_DROP', -22.0, {
    currentRate: 2.8,
    previousRate: 3.6,
    category: 'Electronics',
    affectedProducts: 12
  });
  
  // 4. Advertising Costs
  log('\n4. Marketing & Advertising Signals', 'yellow');
  await sendSignal('AD_COST_SPIKE', 45.0, {
    platform: 'Google Ads',
    campaign: 'Holiday Electronics Sale',
    oldCPC: 12.50,
    newCPC: 18.13,
    budget: 50000
  });
  
  await sendSignal('AD_COST_SPIKE', 30.0, {
    platform: 'Meta Ads',
    campaign: 'Wireless Accessories',
    oldCPC: 8.20,
    newCPC: 10.66,
    budget: 30000
  });
  
  // 5. Shipping Delays
  log('\n5. Logistics & Fulfillment Signals', 'yellow');
  await sendSignal('SHIPPING_DELAY', 3.5, {
    carrier: 'FastShip Express',
    affectedOrders: 87,
    averageDelay: 3.5,
    region: 'Northeast'
  });
  
  // 6. Positive Signals
  log('\n6. Revenue & Growth Signals', 'yellow');
  await sendSignal('REVENUE_SPIKE', 28.0, {
    category: 'Accessories',
    amount: 125000,
    previousAmount: 97656,
    topProduct: 'Phone Case Bundle'
  });
  
  await sendSignal('PRODUCT_TRENDING', 150.0, {
    product: 'Wireless Charging Pad',
    viewIncrease: 150.0,
    salesIncrease: 85.0,
    socialMentions: 340
  });
  
  // 7. Customer Behavior
  log('\n7. Customer Experience Signals', 'yellow');
  await sendSignal('HIGH_RETURN_RATE', 15.5, {
    product: 'Budget Earbuds',
    returnRate: 15.5,
    threshold: 10.0,
    topReason: 'Poor sound quality'
  });
  
  // 8. Additional Activity
  log('\n8. Additional Market Activity', 'yellow');
  await sendSignal('COMPETITOR_PRICE_DROP', -8.0, {
    competitor: 'OnlineElectronics',
    product: 'HDMI Cable',
    oldPrice: 249,
    newPrice: 229
  });
  
  await sendSignal('PRICE_OPTIMIZATION', 5.5, {
    product: 'Laptop Sleeve',
    oldPrice: 599,
    newPrice: 632,
    expectedRevenueIncrease: 12.0
  });
  
  await sendSignal('SEASONAL_DEMAND', 35.0, {
    season: 'Holiday Shopping',
    category: 'Electronics Gifts',
    demandIncrease: 35.0,
    stockLevel: 'Adequate'
  });
  
  // Summary
  log('\n=== Demo Data Generation Complete ===\n', 'green');
  log('Summary:', 'cyan');
  log('  • Generated 15+ realistic signals', 'white');
  log('  • Multiple threat types created', 'white');
  log('  • Revenue and competitor data included', 'white');
  log('  • Dashboard should now show active threats', 'white');
  log('\nNext steps:', 'yellow');
  log('  1. Visit http://localhost:3000', 'cyan');
  log('  2. Check the Threat Feed (left panel)', 'cyan');
  log('  3. View Live Status Bar (top)', 'cyan');
  log('  4. Click on threats for AI attribution analysis', 'cyan');
  log('\nTo re-run this script: node scripts/generate-demo-data.js\n', 'gray');
}

// Run the script
main().catch(error => {
  log(`\n✗ Error: ${error.message}\n`, 'red');
  process.exit(1);
});
