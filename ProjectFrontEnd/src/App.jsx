import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [baseImage, setBaseImage] = useState(null);
  const [compareImages, setCompareImages] = useState([]);
  const [results, setResults] = useState([]);
  const [isComparing, setIsComparing] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [animatingImage, setAnimatingImage] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [processComplete, setProcessComplete] = useState(false);
  const div2Ref = useRef(null);
  const imagesContainerRef = useRef(null);
  const animatingRef = useRef(null);

  const SIMILARITY_THRESHOLD = 0.75;

  // Cleanup function to call backend cleanup endpoint
  const cleanupBackend = async () => {
    try {
      await axios.post("http://localhost:5000/cleanup");
      console.log("Backend cleanup successful");
    } catch (error) {
      console.error("Backend cleanup error:", error);
    }
  };

  const handleSubmit = () => {
    if (!baseImage || compareImages.length === 0) return;
    setResults([]);
    setCurrentImage(null);
    setCurrentResult(null);
    setProcessComplete(false);
    setIsComparing(true);
    // Clean up before starting new comparison
    cleanupBackend();
  };

  const resetAll = () => {
    setBaseImage(null);
    setCompareImages([]);
    setResults([]);
    setCurrentImage(null);
    setCurrentResult(null);
    setIsComparing(false);
    setProcessComplete(false);
    // Clean up backend when resetting
    cleanupBackend();
  };

  const compareSingleImage = async (img) => {
    const formData = new FormData();
    formData.append("reference", baseImage);
    formData.append("images", img);
    try {
      const response = await axios.post("http://localhost:5000/predict", formData);
      const resultData = response.data[0];
      
      // Use the is_defective flag directly from backend if available, or calculate based on similarity score
      const isDefective = resultData.is_defective !== undefined 
        ? resultData.is_defective 
        : (resultData.similarity_score < SIMILARITY_THRESHOLD);
      
      const processedResult = {
        ...resultData,
        similarity: resultData.similarity_score || resultData.similarity || 0,
        defect_percentage: resultData.defect_percentage || 0,
        is_defective: isDefective
      };
      
      // Log result to console
      console.log("Image comparison result:", {
        filename: img.name,
        similarity: processedResult.similarity,
        defect_percentage: processedResult.defect_percentage,
        is_defective: processedResult.is_defective
      });
      
      setResults(prev => [...prev, processedResult]);
      setCurrentResult(processedResult);
      return processedResult;
    } catch (error) {
      console.error("Comparison error:", error);
      const fallbackResult = { 
        similarity: 0, 
        similarity_score: 0,
        defect_percentage: 100,
        is_defective: true,
        prediction: "Different"
      };
      console.log("Error fallback result:", fallbackResult);
      setResults(prev => [...prev, fallbackResult]);
      setCurrentResult(fallbackResult);
      return fallbackResult;
    }
  };

  const animateImage = async () => {
    if (compareImages.length === 0 || animatingImage) return;
    
    const img = compareImages[0];
    setAnimatingImage(img);
    
    const queueItems = document.querySelectorAll('.image-queue');
    if (!queueItems.length) {
      setAnimatingImage(null);
      return;
    }

    const positions = Array.from(queueItems).map(item => {
      const rect = item.getBoundingClientRect();
      return { top: rect.top, left: rect.left, width: rect.width, height: rect.height };
    });


    const animEl = document.createElement('div');
    animEl.className = 'animating-element';
    

    const imgEl = document.createElement('img');
    imgEl.src = URL.createObjectURL(img);
    imgEl.className = 'queue-img';
    animEl.appendChild(imgEl);
    

    document.body.appendChild(animEl);
    animatingRef.current = animEl;

    animEl.style.position = 'fixed';
    animEl.style.zIndex = '1000';
    animEl.style.top = `${positions[0].top}px`;
    animEl.style.left = `${positions[0].left}px`;
    animEl.style.width = `${positions[0].width}px`;
    animEl.style.height = `${positions[0].height}px`;
    animEl.style.borderRadius = '10px';
    animEl.style.overflow = 'hidden';
    animEl.style.boxShadow = '0 0 15px rgba(0, 119, 182, 0.5)';
    
    // Hide the first queue item
    queueItems[0].style.opacity = '0';
    queueItems[0].style.pointerEvents = 'none';
    
    // Prepare other queue items for the upward animation
    for (let i = 1; i < queueItems.length; i++) {
      queueItems[i].style.transition = 'none';
      queueItems[i].style.transform = 'translateY(0)';
    }
    
    // Force browser to register the initial state
    requestAnimationFrame(() => {
      // Start upward animation for remaining items
      for (let i = 1; i < queueItems.length; i++) {
        queueItems[i].style.transition = 'transform 0.8s ease-in-out';
        queueItems[i].style.transform = `translateY(-${positions[0].height + 12}px)`;
      }
      
      // Get target position for the flying element
      const endRect = div2Ref.current.getBoundingClientRect();
      
      // Start animation after a brief delay
      setTimeout(() => {
        animEl.style.transition = 'all 0.8s ease-in-out';
        animEl.style.transform = `translate(${endRect.left - positions[0].left}px, ${endRect.top - positions[0].top}px)`;
        animEl.style.width = `${endRect.width}px`;
        animEl.style.height = `${endRect.height}px`;
      }, 10);
    });
    
    // Wait for animation to complete
    await new Promise(res => setTimeout(res, 850));
    
    // Update current image state
    setCurrentImage(img);
    
    // Remove the animation element
    document.body.removeChild(animEl);
    animatingRef.current = null;
    
    // Process the image comparison
    const result = await compareSingleImage(img);
    
    // Remove processed image from the queue
    setCompareImages(prev => prev.slice(1));
    setAnimatingImage(null);
    
    // Apply highlight effect based on defect detection
    if (result.is_defective) {
      // Apply red shadow effect for defective image
      const imgDiv = div2Ref.current;
      imgDiv.classList.add('defective-highlight');
      setTimeout(() => {
        imgDiv.classList.remove('defective-highlight');
      }, 2000);
    }
  };

  useEffect(() => {
    if (isComparing && !animatingImage && compareImages.length > 0) {
      animateImage();
    } else if (compareImages.length === 0 && results.length > 0) {
      setIsComparing(false);
      setProcessComplete(true);
    }
  }, [isComparing, animatingImage, compareImages.length]);

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      cleanupBackend();
    };
  }, []);

  const toImageURL = (file) => file ? URL.createObjectURL(file) : '';

  // Format similarity value to handle NaN and display as percentage
  const formatSimilarity = (value) => {
    if (typeof value !== 'number' || isNaN(value)) {
      return '0.00%';
    }
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format defect percentage to handle missing values
  const formatDefectPercentage = (value) => {
    if (typeof value !== 'number' || isNaN(value)) {
      return '0.00%';
    }
    return `${value.toFixed(2)}%`;
  };

  const downloadReport = () => {
    // Create report content
    let reportContent = `
      # Defect Detection Report
      Generated on: ${new Date().toLocaleString()}
      
      ## Summary
      Total images processed: ${results.length}
      Defective images found: ${results.filter(r => r.is_defective).length}
      
      ## Detailed Results
      ${results.map((result, index) => `
      ### Image ${index + 1}: ${result.image_name || `Image-${index+1}`}
      Status: ${result.is_defective ? 'DEFECTIVE' : 'NORMAL'}
      Similarity score: ${formatSimilarity(result.similarity)}
      Defect percentage: ${formatDefectPercentage(result.defect_percentage)}
      `).join('\n')}
    `;
    
    // Create and download the file
    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'defect-detection-report.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="main">
      <div className="container">
        <h1>Siamese Defect Detection</h1>
        <div className="input-buttons">
          <input type="file" onChange={(e) => setBaseImage(e.target.files[0])} disabled={isComparing} />
          <input 
            type="file" 
            multiple 
            onChange={(e) => setCompareImages(Array.from(e.target.files))} 
            disabled={isComparing}
          />
        </div>
        <div className="image-divs">
          <div className="div1">
            {baseImage && <img src={toImageURL(baseImage)} alt="Base" className="preview-img" />}
            {!baseImage && <div className="placeholder-text">Reference Image</div>}
          </div>
          <div className="div2" ref={div2Ref}>
            {currentImage && <img src={toImageURL(currentImage)} alt="Current" className="preview-img" />}
            {!currentImage && <div className="placeholder-text">Test Image</div>}
          </div>
        </div>
        
        {currentResult && (
          <div className={`result-display ${currentResult.is_defective ? 'defective' : 'normal'}`}>
            <h3>Result</h3>
            <div className="result-content">
              <div className="result-status">
                Status: <span>{currentResult.is_defective ? 'DEFECTIVE' : 'NORMAL'}</span>
              </div>
              <div className="similarity-score">
                Similarity: <span>{formatSimilarity(currentResult.similarity)}</span>
              </div>
              <div className="defect-percentage">
                Defect %: <span>{formatDefectPercentage(currentResult.defect_percentage)}</span>
              </div>
            </div>
          </div>
        )}
        
        <div className="button-group">
          <button onClick={handleSubmit} disabled={isComparing || !baseImage || compareImages.length === 0}>
            {isComparing ? 'Processing...' : 'Start Simulation'}
          </button>
          
          {processComplete && (
            <>
              <button onClick={downloadReport} className="download-btn">
                Download Report
              </button>
              <button onClick={resetAll} className="reset-btn">
                Reset
              </button>
            </>
          )}
        </div>
      </div>

      <div className="images-container" ref={imagesContainerRef}>
        <div className="queue-header">
          <h3>Image Queue ({compareImages.length})</h3>
        </div>
        {compareImages.map((img, idx) => (
          <div key={idx} className="image-queue">
            <img src={toImageURL(img)} alt={`queue-${idx}`} className="queue-img" />
          </div>
        ))}
        {processComplete && compareImages.length === 0 && results.length > 0 && (
          <div className="queue-complete">
            <div className="complete-message">
              <span className="checkmark">✓</span>
              <p>All images processed!</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;