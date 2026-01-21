# 🛠️ SellerOps Setup Guide

## 1. Environment Infrastructure
We use **Redis** for real-time signal processing and **Turso** for edge database storage.

### ⚡ Start Redis (Required)
Since you are on Windows with Docker installed, start a lightweight Redis instance:

```powershell
# Run Redis container (detached, port 6379)
docker run -d -p 6379:6379 --name seller-ops-redis redis:alpine

# Verify it's running
docker ps
```

*Status: ✅ Verified (Redis is running)*

### 🔑 Environment Keys
Create a `.env` file in the root directory if it doesn't exist:

```properties
# Gemini AI (Intelligence)
GEMINI_API_KEY=your_key_here

# Redis (Real-time Signals)
REDIS_URL=redis://localhost:6379

# Turso (Database)
# Leave blank to use local 'file:local.db' for development
TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=
```

## 2. Run the Application

```powershell
# Install dependencies
npm install

# Run development server
npm run dev
```

### 🗄️ Database Initialization
After starting the dev server, visit this URL once to create the database tables:
http://localhost:3000/api/admin/init

## 3. Testing
We have a robust test suite valid for production:

```powershell
# Run all tests (Unit + component)
npm test

# Run real infrastructure smoke test
npx vitest src/lib/redis/connection.test.ts
```
