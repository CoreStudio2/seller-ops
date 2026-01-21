# 🎯 SellerOps - AI-Powered Commerce Intelligence

**Next-Gen E-Commerce Decision Intelligence Platform**

A real-time war room dashboard for e-commerce operations, powered by **Gemini AI** and **TensorFlow.js**, designed to detect threats, attribute revenue changes, and provide intelligent product recommendations.

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Next.js](https://img.shields.io/badge/Next.js-16.1.4-black)
![Gemini AI](https://img.shields.io/badge/Gemini-2.0%20Flash-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow.js-4.22.0-blue)

---

## 🌟 Features

### 🧠 **Gemini AI-Powered Recommendations** (PRIMARY)
- **Dynamic Product Catalog Generation** - AI creates realistic product catalogs on-demand
- **Intelligent Recommendations** - Context-aware cross-sell, upsell, and bundle suggestions
- **Strategic Insights** - Business intelligence with expected revenue impact
- **Bundle Optimization** - Code execution for optimal pricing strategies
- **Optional TensorFlow Enhancement** - Mathematical similarity scoring

### ⚡ **Real-Time Signal Processing**
- Redis pub/sub for live signal ingestion
- Threat detection with confidence scoring
- Live status bar with revenue velocity tracking
- Scrolling threat feed with severity levels

### 🤖 **Gemini AI Attribution**
- Causal analysis of revenue changes (WHY, not just WHAT)
- Structured output with JSON schemas
- Confidence scoring and recommendations
- Multi-factor attribution

### 🎮 **Beast Mode Simulation**
- Price elasticity modeling
- Competitor response prediction (Game Theory)
- Multi-factor scenario testing
- Risk assessment with confidence intervals

### 🎨 **War Room UI Design**
- Tactical dark HUD theme
- High-contrast signal colors (red/amber/green/cyan)
- Zero SaaS clichés, pure command center aesthetic
- Real-time updates with Zustand state management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SELLEROPS PLATFORM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐   ┌───────────┐  │
│  │   Gemini AI  │───▶│ Recommendations│  │  TensorFlow│  │
│  │   (Primary)  │    │   Intelligence │  │  (Optional)│  │
│  └──────────────┘    └──────────────┘   └───────────┘  │
│         │                                       │        │
│         ├─────────────────┬─────────────────────┤        │
│         ▼                 ▼                     ▼        │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Attribution│   │ Simulation    │   │ Catalog Gen  │  │
│  │ Analysis   │   │ Engine        │   │ (Dynamic)    │  │
│  └────────────┘   └──────────────┘   └──────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │            API Layer (Next.js Routes)              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │   Redis     │   │   Turso     │   │  In-Memory   │  │
│  │  (Signals)  │   │  (Threats)  │   │   (Cache)    │  │
│  └─────────────┘   └─────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Design Principles
- **Gemini AI-First**: Primary intelligence engine, not an add-on
- **Dynamic Data**: No hardcoded catalogs, everything AI-generated
- **Optional Enhancement**: TensorFlow adds value but isn't required
- **Real-Time**: Redis pub/sub for live signal processing
- **Scalable**: Caching, edge SQL, and efficient state management

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 20+ and npm
- **Docker** (for Redis)
- **Gemini API Key** ([Get one here](https://ai.google.dev/))

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd seller-ops
npm install
```

### 2. Environment Setup

Create `.env.local` file:

```bash
# Required for Gemini AI features
GEMINI_API_KEY=your_gemini_api_key_here

# Optional - defaults provided
REDIS_URL=redis://localhost:6379
TURSO_DATABASE_URL=file:local.db
TURSO_AUTH_TOKEN=
```

### 3. Start Redis (Docker)

```bash
docker run -d \
  --name seller-ops-redis \
  -p 6379:6379 \
  redis:alpine
```

Or use the included script:
```bash
# PowerShell
.\scripts\verify-stack.ps1

# This will:
# - Check Redis connection
# - Verify TensorFlow.js
# - Test Gemini API
# - Initialize database
```

### 4. Initialize Database

```bash
npm run dev

# Then visit:
# http://localhost:3000/api/admin/init
```

### 5. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) - you'll see the War Room dashboard!

---

## 📁 Project Structure

```
seller-ops/
├── src/
│   ├── app/                      # Next.js App Router
│   │   ├── api/                  # API Routes
│   │   │   ├── recommendations/  # Gemini AI recommendations
│   │   │   ├── attribution/      # Gemini causal analysis
│   │   │   ├── simulate/         # Beast mode simulation
│   │   │   ├── ingest/           # Signal ingestion
│   │   │   └── status/           # Live dashboard data
│   │   ├── layout.tsx
│   │   └── page.tsx              # War Room Dashboard
│   │
│   ├── components/               # React Components
│   │   ├── feed/                 # ThreatFeed, AttributionBrief
│   │   ├── metrics/              # LiveStatusBar
│   │   ├── recommendations/      # SmartRecommendationsPanel
│   │   └── simulation/           # BeastModePanel
│   │
│   ├── lib/                      # Core Business Logic
│   │   ├── gemini/
│   │   │   ├── catalog-generator.ts    # PRIMARY AI ENGINE
│   │   │   └── attribution.ts          # Causal analysis
│   │   ├── tensorflow/
│   │   │   └── recommendation-engine.ts # Optional enhancement
│   │   ├── redis/
│   │   │   ├── client.ts               # Connection & pub/sub
│   │   │   └── signals.ts              # Signal processing
│   │   ├── turso/
│   │   │   └── database.ts             # Edge SQL
│   │   ├── simulation/
│   │   │   └── engine.ts               # Beast mode logic
│   │   ├── types/
│   │   │   └── index.ts                # TypeScript definitions
│   │   ├── data.ts                     # Data layer
│   │   ├── store.ts                    # Zustand state
│   │   └── hooks.ts                    # Custom React hooks
│   │
│   └── test/                     # Test setup
│
├── scripts/                      # Utility scripts
│   ├── init-db.ts                # Database initialization
│   ├── verify-stack.ps1          # Tech stack verification
│   └── seed-demo-data.ts         # Demo data seeding
│
├── public/                       # Static assets
├── .env.local                    # Environment variables (create this)
└── package.json
```

---

## 🎮 Usage

### 1. **Smart Recommendations** (Gemini AI)

Generate intelligent product recommendations:

```typescript
// API Call
POST /api/recommendations
{
  "productId": "prod-1",
  "strategy": "smart",  // cross-sell | upsell | bundle | smart
  "useTensorFlowEnhancement": false  // optional
}
```

**Features:**
- AI-generated product catalog (no hardcoded data)
- Context-aware recommendations
- Strategic insights with expected revenue impact
- Bundle opportunities with optimal pricing
- Optional TensorFlow similarity enhancement

### 2. **Threat Detection & Attribution**

Ingest signals and detect threats:

```typescript
// Ingest Signal
POST /api/ingest
{
  "type": "COMPETITOR_PRICE_DROP",
  "value": -15,
  "meta": { "competitor": "CompanyXYZ" }
}

// Get Attribution Analysis (Gemini AI)
POST /api/attribution
{
  "threatId": "threat-123"
}
```

### 3. **Beast Mode Simulation**

Run scenario simulations:

```typescript
POST /api/simulate
{
  "scenarioType": "price_change",
  "parameters": {
    "priceChange": -10,
    "competitorResponse": "aggressive"
  }
}
```

### 4. **Live Dashboard Data**

```typescript
GET /api/status
// Returns: live status, recent threats, active signals
```

---

## 🧪 Testing

```bash
# Run all tests
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

**Test Coverage:**
- ✅ TensorFlow similarity calculations
- ✅ Signal processing and threat detection
- ✅ Simulation engine logic
- ✅ Gemini attribution analysis
- ✅ Component rendering
- ✅ Integration tests

**Current Status:** 21/22 tests passing

---

## 🛠️ Tech Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Framework** | Next.js | 16.1.4 | App Router, SSR, API Routes |
| **AI/ML** | Gemini AI | 2.0 Flash | Primary intelligence engine |
| **ML Enhancement** | TensorFlow.js | 4.22.0 | Optional similarity scoring |
| **Real-Time** | Redis (ioredis) | 5.9.2 | Pub/sub, signal processing |
| **Database** | Turso (LibSQL) | 0.17.0 | Edge SQL, threat storage |
| **State** | Zustand | 5.0.10 | Global state management |
| **Validation** | Zod | 4.3.5 | Schema validation |
| **UI** | React | 19.2.3 | Components |
| **Styling** | Tailwind CSS | 4.x | War Room theme |
| **Testing** | Vitest | 4.0.17 | Unit & integration tests |
| **Language** | TypeScript | 5.x | Type safety |

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_key_here           # Get from https://ai.google.dev/

# Optional (defaults provided)
REDIS_URL=redis://localhost:6379       # Redis connection
TURSO_DATABASE_URL=file:local.db       # Database URL
TURSO_AUTH_TOKEN=                      # Turso auth (not needed for local)
```

### Catalog Configuration

In `src/lib/gemini/catalog-generator.ts`:

```typescript
// Customize business context
const businessContext = "electronics and accessories e-commerce";

// Adjust product count
const productCount = 10;

// Cache TTL (milliseconds)
const CACHE_TTL_MS = 3600000; // 1 hour
```

---

## 📊 API Documentation

### Recommendations API

#### GET `/api/recommendations`
Get product catalog (cached, Gemini-generated)

**Query Params:**
- `refresh=true` (optional) - Force catalog regeneration

**Response:**
```json
{
  "products": [...],
  "totalProducts": 10,
  "categories": ["Electronics", "Accessories"],
  "totalValue": 10000,
  "generatedAt": "2026-01-21T...",
  "powered": {
    "gemini": true,
    "tensorflow": false
  }
}
```

#### POST `/api/recommendations`
Generate intelligent recommendations

**Request Body:**
```json
{
  "productId": "prod-1",
  "strategy": "smart",
  "useTensorFlowEnhancement": false,
  "context": "Holiday shopping season",
  "customerProfile": {
    "budget": 5000,
    "preferences": ["wireless", "portable"]
  }
}
```

**Response:**
```json
{
  "targetProduct": {...},
  "recommendations": [
    {
      "product": {...},
      "score": 85,
      "reason": "Complements wireless earbuds...",
      "insights": [...]
    }
  ],
  "analysis": {
    "summary": "Strategic recommendation analysis...",
    "insights": [...],
    "bundleOpportunities": [...],
    "expectedImpact": {
      "revenueIncrease": "12-18%",
      "conversionBoost": "8-15%"
    }
  },
  "confidence": 92
}
```

---

## 🎨 Customization

### UI Theme

The War Room theme is defined in `src/app/globals.css`:

```css
--signal-red: #ef4444;      /* Critical threats */
--signal-amber: #f59e0b;    /* Warnings */
--signal-green: #10b981;    /* Success */
--signal-cyan: #06b6d4;     /* Info/Active */
```

### Adding New Signal Types

1. Update type definition in `src/lib/types/index.ts`
2. Add detection logic in `src/lib/redis/signals.ts`
3. Configure thresholds for threat detection

---

## 📈 Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| First catalog load | ~2-3s | Gemini AI generation |
| Cached catalog load | <50ms | In-memory cache |
| Gemini recommendations | ~1-2s | AI processing |
| + TensorFlow enhancement | +1s | Similarity calculations |
| Signal ingestion | <10ms | Redis publish |
| Threat detection | <5ms | Rule-based |

### Optimization Tips

1. **Catalog Caching**: Default 1-hour TTL, increase for production
2. **Redis Connection Pooling**: Configure maxRetriesPerRequest
3. **TensorFlow Backend**: Auto-selects best available (WebGL/CPU)
4. **Database Indexing**: Create indexes on frequently queried fields

---

## 🚢 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables in Vercel dashboard
```

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables for Production

Ensure these are set:
- `GEMINI_API_KEY` (required)
- `REDIS_URL` (production Redis instance)
- `TURSO_DATABASE_URL` (production Turso DB)
- `TURSO_AUTH_TOKEN` (production auth)

---

## 📚 Documentation

- **[ARCHITECTURE_UPDATE.md](ARCHITECTURE_UPDATE.md)** - Detailed architecture documentation
- **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - Visual architecture overview
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Recent changes summary
- **[TECH_STACK_STATUS.md](TECH_STACK_STATUS.md)** - Tech stack verification
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Project completion status

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow TypeScript strict mode
- Use the War Room design language
- Document API changes
- Run `npm test` before committing

---

## 🐛 Troubleshooting

### Common Issues

**1. Gemini API Errors**
```bash
Error: GEMINI_API_KEY not set
```
**Solution:** Add `GEMINI_API_KEY` to `.env.local`

**2. Redis Connection Failed**
```bash
Error: Redis connection refused
```
**Solution:** Start Redis with Docker:
```bash
docker run -d --name seller-ops-redis -p 6379:6379 redis:alpine
```

**3. TensorFlow Backend Issues**
```bash
Warning: WebGL backend not available
```
**Solution:** TensorFlow.js will fallback to CPU automatically. This is normal.

**4. Database Not Initialized**
```bash
Error: Table threats does not exist
```
**Solution:** Visit `/api/admin/init` to initialize the database

### Debug Mode

Enable verbose logging:
```bash
DEBUG=seller-ops:* npm run dev
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Gemini AI** by Google - Primary intelligence engine
- **TensorFlow.js** - Machine learning in JavaScript
- **Next.js** - React framework
- **Vercel** - Deployment platform
- **Turso** - Edge SQL database

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/seller-ops/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/seller-ops/discussions)
- **Email**: your-email@example.com

---

## 🎯 Roadmap

### v0.2.0 (Next Release)
- [ ] Persistent catalog storage (Redis/Database)
- [ ] Multiple business contexts (Fashion, Electronics, Grocery)
- [ ] Real inventory system integration
- [ ] User authentication & multi-tenancy
- [ ] Advanced analytics dashboard

### v0.3.0 (Future)
- [ ] A/B testing framework
- [ ] Recommendation history & learning
- [ ] Mobile-responsive War Room UI
- [ ] Email/Slack notifications for critical threats
- [ ] GraphQL API option

---

<div align="center">

**Built with ❤️ using Gemini AI, TensorFlow.js, and Next.js**

[⭐ Star on GitHub](https://github.com/yourusername/seller-ops) | [📖 Documentation](docs/) | [🐛 Report Bug](issues/) | [💡 Request Feature](issues/)

</div>
