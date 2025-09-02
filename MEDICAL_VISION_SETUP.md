# Google Medical Vision AI Setup Guide

## 🎯 What This Does

The **Google Medical Vision AI** integration provides **intelligent medical image analysis** for creating better image occlusion flashcards. Instead of blindly masking random text regions, it:

- **Understands medical content** (anatomy, drugs, measurements, diagnoses)
- **Identifies testable regions** based on medical education value
- **Scores regions intelligently** using medical knowledge
- **Provides rationale** for why each region is good for testing

## 🚀 Benefits Over Basic Tesseract

| Feature | Tesseract (Basic) | Google Medical Vision AI |
|---------|-------------------|-------------------------|
| **Text Detection** | ✅ Good | ✅ Excellent |
| **Medical Understanding** | ❌ None | ✅ Advanced |
| **Testing Value** | ❌ Random | ✅ Intelligent |
| **Region Scoring** | ❌ Basic | ✅ Medical-aware |
| **Rationale** | ❌ None | ✅ Detailed |

## 📋 Prerequisites

1. **Google Cloud Account** (free tier available)
2. **Google Cloud Vision API** enabled
3. **Service Account** with Vision API permissions
4. **Python packages** installed

## 🔧 Setup Steps

### Step 1: Enable Google Cloud Vision API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Cloud Vision API**
4. Go to APIs & Services → Library → Search "Vision API" → Enable

### Step 2: Create Service Account

1. Go to **IAM & Admin** → **Service Accounts**
2. Click **Create Service Account**
3. Name: `medical-vision-ai`
4. Description: `Service account for medical image analysis`
5. Click **Create and Continue**

### Step 3: Grant Permissions

1. **Role**: `Cloud Vision API User`
2. Click **Continue** → **Done**

### Step 4: Download Credentials

1. Click on your service account
2. Go to **Keys** tab
3. Click **Add Key** → **Create New Key**
4. Choose **JSON** format
5. Download the file and save as `google_credentials.json`

### Step 5: Install Dependencies

```bash
# Install Google Cloud Vision package
pip install google-cloud-vision

# Or update requirements.txt and run
pip install -r requirements.txt
```

### Step 6: Configure Environment

```bash
# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_credentials.json"

# Or add to your .env file
echo "GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google_credentials.json" >> .env
```

### Step 7: Update Config

In `config.yaml`, ensure:

```yaml
image_occlusion:
  use_google_vision: true
  google_credentials_path: "/path/to/your/google_credentials.json"
```

## 🧪 Testing the Setup

Run the test script:

```bash
python test_medical_vision.py
```

You should see:
```
🧪 Testing Google Medical Vision AI Provider...
✅ Google Medical Vision AI is available!
✅ Provider created successfully
✅ Client initialized: True
```

## 💰 Cost Information

- **Google Cloud Vision API**: $1.50 per 1000 images
- **Free Tier**: 1000 requests/month
- **Typical Use**: 1-5 images per PowerPoint = ~$0.002-0.008 per deck

## 🎯 How It Works

### 1. **Image Analysis**
- Text detection with bounding boxes
- Medical object recognition
- Content classification

### 2. **Intelligent Scoring**
- **Anatomical labels**: +0.4 points
- **Measurements**: +0.3 points  
- **Drug names**: +0.35 points
- **Diagnoses**: +0.3 points
- **Position**: +0.2 points (center = better)
- **Length**: +0.15 points (optimal = 3-15 chars)

### 3. **Region Selection**
- Filters by quality threshold (≥0.3)
- Ensures diversity of region types
- Prioritizes medical relevance

### 4. **Fallback Strategy**
- If Google Vision fails → Tesseract
- If Tesseract fails → Basic region detection
- Always generates some masks

## 🔍 Example Output

```
[INFO] Using Google Medical Vision AI for intelligent medical image analysis
[INFO] Medical Vision AI found 4 testable regions in slide5_img2.jpg
  Region 1: "Aorta" (Score: 0.85)
    Type: anatomical_label
    Rationale: Anatomical labels are excellent for testing; Centrally positioned text is often more important
  Region 2: "120/80" (Score: 0.72)
    Type: measurement
    Rationale: Numerical values and measurements are good test targets
```

## 🚨 Troubleshooting

### "Google Medical Vision AI Not Available"

1. **Check credentials**: `echo $GOOGLE_APPLICATION_CREDENTIALS`
2. **Verify file exists**: `ls -la /path/to/google_credentials.json`
3. **Check permissions**: File should be readable
4. **Verify API enabled**: Check Google Cloud Console

### "Import Error: google.cloud.vision"

```bash
pip install google-cloud-vision
```

### "Authentication Error"

1. Verify service account has Vision API permissions
2. Check if credentials file is valid JSON
3. Ensure project has billing enabled

## 🎉 What You Get

With Google Medical Vision AI enabled:

- **Smarter masking** - Only covers medically relevant text
- **Better test questions** - Focuses on important content
- **Medical context** - Understands anatomy, drugs, measurements
- **Quality assurance** - Scores regions by testing value
- **Professional results** - Medical education-grade flashcards

## 🔄 Fallback Behavior

The system automatically falls back to Tesseract if Google Vision fails, ensuring you always get image occlusion flashcards, just with less intelligent masking.

---

**Ready to create intelligent medical flashcards?** 🏥✨










