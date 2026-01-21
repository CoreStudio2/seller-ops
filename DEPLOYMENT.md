# SellerOps Vercel Deployment Guide

## 🚀 Quick Deployment

### **Prerequisites**
1. GitHub account
2. Vercel account (sign up at vercel.com)
3. Gemini API key
4. Upstash Redis account (free tier)

---

## Step 1: Set Up Upstash Redis

1. Visit https://upstash.com
2. Sign up / Login
3. Click "Create Database"
   - Name: `seller-ops-redis`
   - Region: Choose closest to your users
   - Type: Regional (free tier)
4. Copy credentials:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`

---

## Step 2: Prepare Local Environment

```bash
# Copy environment template
cp .env.example .env.local

# Edit .env.local and add:
# - GEMINI_API_KEY
# - UPSTASH_REDIS_REST_URL
# - UPSTASH_REDIS_REST_TOKEN
```

---

## Step 3: Test Build Locally

```bash
# Install dependencies
npm install

# Build for production
npm run build

# Test production build
npm start
```

Visit http://localhost:3000 and verify:
- ✅ All pages load
- ✅ Charts render
- ✅ API routes work
- ✅ No console errors

---

## Step 4: Push to GitHub

```bash
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit - ready for Vercel"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/seller-ops.git
git branch -M main
git push -u origin main
```

---

## Step 5: Deploy to Vercel

### **Option A: Vercel Dashboard (Recommended)**

1. Visit https://vercel.com/new
2. Import your GitHub repository
3. Configure project:
   - Framework Preset: **Next.js**
   - Root Directory: `./`
   - Build Command: `npm run build`
   - Output Directory: `.next`
4. Add Environment Variables:
   ```
   GEMINI_API_KEY=your_key
   UPSTASH_REDIS_REST_URL=your_url
   UPSTASH_REDIS_REST_TOKEN=your_token
   ```
5. Click "Deploy"

### **Option B: Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# Follow prompts, then add env vars via dashboard
# Deploy to production
vercel --prod
```

---

## Step 6: Verify Deployment

After deployment, test your production URL:

1. **Homepage** - Should load with all UI elements
2. **Analytics Tab** - All 6 charts should render
3. **Mobile View** - Resize browser, test hamburger menu
4. **API Routes:**
   - `/api/status` - Returns live status
   - `/api/recommendations` - Returns Gemini recommendations
5. **Console** - No errors in browser console

---

## 📋 Post-Deployment Checklist

- [ ] Custom domain configured (optional)
- [ ] All environment variables set
- [ ] Preview deployments working
- [ ] Mobile responsiveness verified
- [ ] Performance metrics checked (Vercel Analytics)
- [ ] Error monitoring set up
- [ ] README updated with live URL

---

## ⚙️ Vercel Configuration

### **vercel.json** (Optional)

```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "regions": ["iad1"],
  "functions": {
    "src/app/api/**/*.ts": {
      "maxDuration": 10
    }
  }
}
```

---

## 🐛 Troubleshooting

### **Issue: "Module not found" error**
**Solution:** Ensure all imports use correct paths. Check `tsconfig.json` paths.

### **Issue: Charts not rendering**
**Solution:** Charts use dynamic imports and client-side rendering. Check browser console for errors.

### **Issue: Redis connection failed**
**Solution:**
1. Verify Upstash credentials in Vercel dashboard
2. Check Redis mode in logs: `/api/status` should show `"redisMode": "upstash"`
3. Fallback mode will work even without Redis

### **Issue: Gemini API timeout**
**Solution:**
1. Check API key is correct
2. Consider upgrading to Vercel Pro (60s timeout vs 10s)
3. Or optimize Gemini API calls

### **Issue: Build fails**
**Solution:**
```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules
npm install

# Try build again
npm run build
```

---

## 📊 Monitoring

### **Vercel Analytics**
- Enable in: Project Settings → Analytics
- View metrics: Dashboard → Analytics tab

### **Redis Monitoring**
- Upstash Dashboard shows:
  - Request count
  - Database size
  - Latency

### **Error Tracking**
- Check: Vercel Dashboard → Functions → Logs
- Filter by: Errors only

---

## 🔄 Continuous Deployment

Vercel auto-deploys on:
- **Push to `main`** → Production
- **Push to other branches** → Preview deployments
- **Pull requests** → Preview URLs

---

## 💰 Cost Estimate

### **Free Tier (Hobby)**
- Vercel: 100 GB bandwidth/month
- Upstash: 10K requests/day
- **Total: $0/month**

### **If You Exceed Free Tier**
- Vercel Pro: $20/month (60s functions, 1TB bandwidth)
- Upstash Pro: $10/month (higher limits)

---

## 📱 Custom Domain (Optional)

1. Buy domain (Namecheap, GoDaddy, etc.)
2. Vercel Dashboard → Project → Settings → Domains
3. Add your domain
4. Update DNS records (Vercel provides instructions)
5. SSL automatically provisioned

---

## ✅ Success!

Your SellerOps dashboard is now live at:
**https://your-project.vercel.app**

Share the URL and enjoy your production deployment! 🎉

---

## 📞 Support

- **Vercel Docs:** https://vercel.com/docs
- **Upstash Docs:** https://docs.upstash.com
- **Next.js Docs:** https://nextjs.org/docs
