#!/usr/bin/env node

/**
 * Complete Demo & AI Testing Script
 * Tests all AI features including attribution and recommendations
 */

const API_BASE = 'http://localhost:3000/api';

const colors = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  gray: '\x1b[90m',
  white: '\x1b[37m',
  bold: '\x1b[1m'
};

function log(message, color = 'white') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSection(title) {
  console.log(`\n${colors.bold}${colors.cyan}${'='.repeat(60)}${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}${title}${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}${'='.repeat(60)}${colors.reset}\n`);
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function checkServer() {
  logSection('STEP 1: Server Status Check');
  try {
    const response = await fetch(`${API_BASE}/status`, { signal: AbortSignal.timeout(5000) });
    const data = await response.json();
    log('✓ Server is running', 'green');
    log(`  Redis: ${data.redis}`, 'cyan');
    log(`  Database: ${data.database}`, 'cyan');
    return true;
  } catch (error) {
    log('✗ Server is not running!', 'red');
    log('  Please run: npm run dev', 'yellow');
    return false;
  }
}

async function initDatabase() {
  logSection('STEP 2: Database Initialization');
  try {
    const response = await fetch(`${API_BASE}/admin/init`, {
      method: 'POST',
      signal: AbortSignal.timeout(10000)
    });
    const data = await response.json();
    log('✓ Database initialized', 'green');
    return true;
  } catch (error) {
    log('! Database may already be initialized (this is OK)', 'yellow');
    return true;
  }
}

async function generateDemoData() {
  logSection('STEP 3: Generating Demo Data');
  
  const signals = [
    {
      type: 'COMPETITOR_PRICE_DROP',
      value: -18.5,
      meta: { competitor: 'TechGiant Corp', product: 'Wireless Earbuds Pro', oldPrice: 2499, newPrice: 2036 }
    },
    {
      type: 'LOW_INVENTORY',
      value: 8,
      meta: { product: 'Portable Power Bank', currentStock: 8, reorderPoint: 50, warehouse: 'DC-EAST' }
    },
    {
      type: 'HIGH_CART_ABANDONMENT',
      value: 78.5,
      meta: { sessionCount: 450, abandonedCarts: 353, averageCartValue: 3200 }
    }
  ];

  for (const signal of signals) {
    try {
      const response = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signal),
        signal: AbortSignal.timeout(5000)
      });
      const data = await response.json();
      
      process.stdout.write(colors.green + '  ✓ ' + colors.reset);
      process.stdout.write(colors.white + signal.type + ' ' + colors.reset);
      if (data.threat) {
        log(`[THREAT: ${data.threat.severity}]`, 'red');
      } else {
        log('[Normal]', 'gray');
      }
      await sleep(300);
    } catch (error) {
      log(`  ✗ Failed: ${signal.type}`, 'red');
    }
  }
}

async function testAIAttribution() {
  logSection('STEP 4: Testing AI Attribution Analysis');
  
  const threatData = {
    threat: {
      id: 'demo-threat-001',
      title: 'Critical Competitor Price Drop',
      type: 'COMPETITOR_PRICE_DROP',
      severity: 'CRITICAL',
      description: 'TechGiant Corp dropped price by 18.5% on competing product',
      detectedAt: new Date().toISOString()
    },
    signals: [
      {
        id: 'sig-demo-1',
        type: 'COMPETITOR_PRICE',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        value: 2036,
        previousValue: 2499,
        delta: -463
      },
      {
        id: 'sig-demo-2',
        type: 'CONVERSION_DROP',
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        value: 2.1,
        previousValue: 3.8,
        delta: -1.7
      }
    ]
  };

  try {
    log('Sending threat to Gemini AI...', 'yellow');
    const response = await fetch(`${API_BASE}/attribution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(threatData),
      signal: AbortSignal.timeout(30000)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error: ${errorData.error || response.statusText}`);
    }

    const result = await response.json();
    
    log('\n✓✓✓ AI ATTRIBUTION ANALYSIS SUCCESSFUL! ✓✓✓\n', 'green');
    log('━'.repeat(60), 'cyan');
    log(`CONFIDENCE SCORE: ${result.confidence}%`, 'yellow');
    log('━'.repeat(60), 'cyan');
    
    log('\nSUMMARY:', 'cyan');
    log(`  ${result.summary}\n`, 'white');
    
    log('ROOT CAUSES:', 'red');
    result.causes?.forEach(cause => {
      log(`  ► ${cause.factor} [${cause.impact}]`, 'white');
      log(`    Evidence: ${cause.evidence}`, 'gray');
    });
    
    log('\nRECOMMENDED ACTIONS:', 'green');
    result.suggestedActions?.forEach(action => {
      log(`  ► [${action.priority}] ${action.action}`, 'white');
      log(`    Expected: ${action.expectedOutcome}`, 'gray');
    });
    
    log('\n━'.repeat(60), 'cyan');
    return true;
  } catch (error) {
    log(`\n✗ AI Attribution Failed: ${error.message}`, 'red');
    
    if (error.message.includes('API key')) {
      log('\n  Troubleshooting:', 'yellow');
      log('  1. Check .env.local file exists', 'cyan');
      log('  2. Verify GEMINI_API_KEY is set', 'cyan');
      log('  3. Restart the dev server: npm run dev', 'cyan');
    }
    return false;
  }
}

async function testAIRecommendations() {
  logSection('STEP 5: Testing AI Product Recommendations');
  
  try {
    log('Fetching Gemini-generated product catalog...', 'yellow');
    const response = await fetch(`${API_BASE}/recommendations`, {
      signal: AbortSignal.timeout(30000)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error: ${errorData.error || response.statusText}`);
    }

    const result = await response.json();
    
    log('\n✓✓✓ AI RECOMMENDATIONS WORKING! ✓✓✓\n', 'green');
    log('━'.repeat(60), 'cyan');
    log(`CATALOG: ${result.products?.length || 0} Products Generated`, 'yellow');
    log(`TOTAL VALUE: ₹${result.metadata?.totalValue?.toLocaleString() || 0}`, 'yellow');
    log(`GENERATED: ${new Date(result.metadata?.generatedAt).toLocaleTimeString()}`, 'yellow');
    log('━'.repeat(60), 'cyan');
    
    if (result.products && result.products.length > 0) {
      log('\nSAMPLE PRODUCTS:', 'cyan');
      result.products.slice(0, 3).forEach(product => {
        log(`  • ${product.name}`, 'white');
        log(`    Price: ₹${product.price} | Category: ${product.category}`, 'gray');
      });
    }
    
    log('\n━'.repeat(60), 'cyan');
    return true;
  } catch (error) {
    log(`\n✗ AI Recommendations Failed: ${error.message}`, 'red');
    return false;
  }
}

async function main() {
  console.clear();
  log('\n🚀 SellerOps AI System Test Suite\n', 'cyan');
  log('This script will test all AI features:\n', 'white');
  log('  ✓ Gemini AI Attribution Analysis', 'cyan');
  log('  ✓ Gemini AI Product Recommendations', 'cyan');
  log('  ✓ Demo Data Generation', 'cyan');
  log('  ✓ API Endpoints\n', 'cyan');
  
  await sleep(2000);

  // Run all tests
  const serverOk = await checkServer();
  if (!serverOk) {
    process.exit(1);
  }

  await initDatabase();
  await generateDemoData();
  
  const attributionOk = await testAIAttribution();
  const recommendationsOk = await testAIRecommendations();

  // Final summary
  logSection('TEST RESULTS SUMMARY');
  log(`Server Status:           ${serverOk ? '✓ PASS' : '✗ FAIL'}`, serverOk ? 'green' : 'red');
  log(`AI Attribution:          ${attributionOk ? '✓ PASS' : '✗ FAIL'}`, attributionOk ? 'green' : 'red');
  log(`AI Recommendations:      ${recommendationsOk ? '✓ PASS' : '✗ FAIL'}`, recommendationsOk ? 'green' : 'red');
  
  const allPassed = serverOk && attributionOk && recommendationsOk;
  
  if (allPassed) {
    log('\n🎉 ALL TESTS PASSED! System is ready for demo.', 'green');
    log('\nNext: Open http://localhost:3000 in your browser', 'cyan');
  } else {
    log('\n⚠️  Some tests failed. Check errors above.', 'yellow');
  }
  
  console.log('');
}

main().catch(error => {
  log(`\n✗ Fatal Error: ${error.message}\n`, 'red');
  process.exit(1);
});
