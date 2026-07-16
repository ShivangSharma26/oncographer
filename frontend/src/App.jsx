import { useState, useCallback } from 'react'
import { UploadCloud, Activity, LayoutDashboard, AlertTriangle, CheckCircle, RefreshCcw } from 'lucide-react'

function App() {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const processFile = async (selectedFile) => {
    setFile(selectedFile)
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      // Pointing to FastAPI backend running on port 8000
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process image. Make sure FastAPI is running.')
      }

      const data = await response.json()
      setResult({
        ...data,
        originalImage: URL.createObjectURL(selectedFile)
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Premium Header */}
      <header className="glass-panel border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-red-600 to-red-900 flex items-center justify-center glow-red">
              <Activity className="text-white w-6 h-6" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Oncographer <span className="text-red-500 font-light">AI</span>
            </h1>
          </div>
          
          <a 
            href="http://127.0.0.1:5000" 
            target="_blank" 
            rel="noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-sm font-medium hover:glow-red"
          >
            <LayoutDashboard className="w-4 h-4 text-red-500" />
            MLflow Dashboard
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-12 flex flex-col items-center justify-center">
        
        {/* Title Section */}
        {!result && !loading && (
          <div className="text-center mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Upload Tissue Patch
            </h2>
            <p className="text-gray-400 max-w-xl mx-auto text-lg">
              Drag and drop a histopathology patch to instantly detect cancerous regions using our fine-tuned ResNet18 model with Grad-CAM visualization.
            </p>
          </div>
        )}

        {/* Upload Zone */}
        {!result && !loading && (
          <div className="w-full max-w-2xl animate-in fade-in zoom-in-95 duration-500 delay-150">
            <div 
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative overflow-hidden rounded-3xl glass-panel border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-16 text-center cursor-pointer
                ${dragActive ? 'border-red-500 bg-red-500/5 glow-red' : 'border-white/20 hover:border-red-500/50 hover:bg-white/5'}
              `}
            >
              <input 
                type="file" 
                accept="image/png, image/jpeg, image/jpg" 
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              
              <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6">
                <UploadCloud className={`w-10 h-10 ${dragActive ? 'text-red-500' : 'text-gray-400'}`} />
              </div>
              <h3 className="text-2xl font-semibold mb-2">Drop your image here</h3>
              <p className="text-gray-500">or click to browse local files</p>
              <p className="text-xs text-gray-600 mt-6">Supports .PNG, .JPG up to 10MB</p>
            </div>
            
            {error && (
              <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <p>{error}</p>
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 animate-in fade-in">
            <div className="relative w-24 h-24 mb-8">
              <div className="absolute inset-0 border-t-2 border-red-500 rounded-full animate-spin"></div>
              <div className="absolute inset-2 border-r-2 border-white/20 rounded-full animate-spin-reverse"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <Activity className="w-8 h-8 text-red-500 animate-pulse" />
              </div>
            </div>
            <h3 className="text-xl font-medium tracking-wide">Analyzing Tissue Patch...</h3>
            <p className="text-gray-500 mt-2">Computing Grad-CAM gradients</p>
          </div>
        )}

        {/* Results Dashboard */}
        {result && !loading && (
          <div className="w-full animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-3xl font-bold">Analysis Complete</h2>
              <button 
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm font-medium"
              >
                <RefreshCcw className="w-4 h-4" />
                Analyze Another
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Score Panel */}
              <div className="glass-panel rounded-3xl p-8 flex flex-col justify-between relative overflow-hidden">
                <div className={`absolute top-0 left-0 w-1 h-full ${result.is_suspicious ? 'bg-red-500' : 'bg-green-500'}`}></div>
                
                <div>
                  <div className="flex items-center gap-3 mb-6">
                    {result.is_suspicious ? (
                      <AlertTriangle className="w-8 h-8 text-red-500" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    )}
                    <h3 className="text-xl font-medium text-gray-400">Diagnosis</h3>
                  </div>
                  
                  <div className="mb-2">
                    <span className="text-6xl font-bold tracking-tighter">
                      {(result.cancer_probability * 100).toFixed(1)}<span className="text-3xl text-gray-500">%</span>
                    </span>
                  </div>
                  <p className="text-gray-400 text-lg">Probability of Cancer</p>
                </div>
                
                <div className={`mt-8 inline-flex px-4 py-2 rounded-full text-sm font-semibold border
                  ${result.is_suspicious 
                    ? 'bg-red-500/10 border-red-500/30 text-red-400 glow-red' 
                    : 'bg-green-500/10 border-green-500/30 text-green-400'
                  }
                `}>
                  {result.is_suspicious ? 'HIGHLY SUSPICIOUS' : 'BENIGN / HEALTHY'}
                </div>
              </div>

              {/* Images Panel */}
              <div className="lg:col-span-2 glass-panel rounded-3xl p-8">
                <h3 className="text-xl font-medium text-gray-400 mb-6">Visual Explanations</h3>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                  {/* Original */}
                  <div className="flex flex-col gap-3">
                    <div className="aspect-square rounded-2xl overflow-hidden border border-white/10 bg-black/50 relative">
                      <img src={result.originalImage} alt="Original Patch" className="w-full h-full object-cover" />
                    </div>
                    <p className="text-center text-sm font-medium text-gray-400">Original Patch</p>
                  </div>
                  
                  {/* Heatmap */}
                  <div className="flex flex-col gap-3">
                    <div className="aspect-square rounded-2xl overflow-hidden border border-red-500/30 bg-black/50 relative glow-red">
                      <img 
                        src={`data:image/png;base64,${result.gradcam_heatmap_base64}`} 
                        alt="Grad-CAM Heatmap" 
                        className="w-full h-full object-cover mix-blend-screen"
                      />
                    </div>
                    <p className="text-center text-sm font-medium text-red-400">Grad-CAM Heatmap</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  )
}

export default App
