import React, { useState, useRef, useEffect } from 'react';
import './index.css';

interface Prediction {
  breed: string;
  probability: number;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useSteganography, setUseSteganography] = useState(false);
  const [message, setMessage] = useState('');
  const [savePath, setSavePath] = useState('');
  const [encodedImage, setEncodedImage] = useState<string | null>(null);
  const [decodedMessage, setDecodedMessage] = useState<string | null>(null);
  const [saveToServer, setSaveToServer] = useState(false);
  const [serverPath, setServerPath] = useState<string | null>(null);
  const [stegMode, setStegMode] = useState<'encode' | 'decode'>('encode');
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [processedImageBlob, setProcessedImageBlob] = useState<Blob | null>(null);
  const [selectedFilter, setSelectedFilter] = useState('');
  const [filterParams, setFilterParams] = useState<Record<string, any>>({});
  const [availableFilters, setAvailableFilters] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setPredictions([]);
      setError(null);
      setEncodedImage(null);
      setDecodedMessage(null);
    }
  };

  const handleEncode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !message) return;

    setLoading(true);
    setError(null);
    setEncodedImage(null);
    setServerPath(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('message', message);
    if (savePath) {
      formData.append('save_path', savePath);
    }
    formData.append('save_to_server', saveToServer.toString());

    try {
      const response = await fetch('http://localhost:8000/encode', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to encode message');
      }

      const data = await response.json();
      setEncodedImage(data.encoded_image);
      if (data.server_path) {
        setServerPath(data.server_path);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDecode = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setDecodedMessage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/decode', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to decode message');
      }

      const data = await response.json();
      setDecodedMessage(data.message);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleProcessImage = async () => {
    if (!file || !selectedFilter) return;

    setLoading(true);
    setError(null);
    setProcessedImage(null);
    setProcessedImageBlob(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('filter_type', selectedFilter);
    if (Object.keys(filterParams).length > 0) {
      formData.append('params', JSON.stringify(filterParams));
    }

    try {
      const response = await fetch('http://localhost:8000/process-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image');
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setProcessedImage(imageUrl);
      setProcessedImageBlob(blob);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setPredictions([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image');
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch('http://localhost:8000/available-filters')
      .then(res => res.json())
      .then(data => setAvailableFilters(data.filters))
      .catch(err => console.error('Error fetching filters:', err));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Dog Breed Classifier with Steganography</h1>
        <p>Upload an image to classify dog breeds, encode/decode messages, and process images with filters</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-input-container">
            {(!useSteganography || stegMode) && (
              <>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="file-input"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="upload-button"
                >
                  Choose Image
                </button>
                {file && <span className="file-name">{file.name}</span>}
              </>
            )}
          </div>

          {/* Filter controls UI */}
          {file && (
            <div className="filter-controls">
              <h3>Image Processing</h3>
              <select
                value={selectedFilter}
                onChange={(e) => setSelectedFilter(e.target.value)}
                className="filter-select"
              >
                <option value="">Select a filter</option>
                {availableFilters.map((filter: string) => (
                  <option key={filter} value={filter}>
                    {filter.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </option>
                ))}
              </select>

              {/* Parameter controls for filters */}
              {selectedFilter === 'brightness' && (
                <div>
                  <label>Brightness Factor:</label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={filterParams.factor ?? 1}
                    onChange={(e) => setFilterParams({ ...filterParams, factor: parseFloat(e.target.value) })}
                  />
                </div>
              )}
              {selectedFilter === 'contrast' && (
                <div>
                  <label>Contrast Factor:</label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={filterParams.factor ?? 1}
                    onChange={(e) => setFilterParams({ ...filterParams, factor: parseFloat(e.target.value) })}
                  />
                </div>
              )}
              {selectedFilter === 'saturation' && (
                <div>
                  <label>Saturation Factor:</label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={filterParams.factor ?? 1}
                    onChange={(e) => setFilterParams({ ...filterParams, factor: parseFloat(e.target.value) })}
                  />
                </div>
              )}
              {selectedFilter === 'rotate' && (
                <div>
                  <label>Rotation Angle:</label>
                  <input
                    type="range"
                    min="0"
                    max="360"
                    step="1"
                    value={filterParams.angle ?? 0}
                    onChange={(e) => setFilterParams({ ...filterParams, angle: parseInt(e.target.value) })}
                  />
                </div>
              )}

              <button
                type="button"
                onClick={handleProcessImage}
                disabled={!selectedFilter || loading}
                className="process-button"
              >
                {loading ? 'Processing...' : 'Process Image'}
              </button>

              {processedImage && (
                <div className="processed-image">
                  <h4>Processed Image:</h4>
                  <img src={processedImage} alt="Processed" />
                </div>
              )}
            </div>
          )}

          <div className="steganography-options">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useSteganography}
                onChange={(e) => setUseSteganography(e.target.checked)}
              />
              Use Steganography
            </label>

            {useSteganography && (
              <div className="steganography-mode">
                <label>
                  <input
                    type="radio"
                    name="stegMode"
                    value="encode"
                    checked={stegMode === 'encode'}
                    onChange={() => setStegMode('encode')}
                  />
                  Encode
                </label>
                <label style={{ marginLeft: '1rem' }}>
                  <input
                    type="radio"
                    name="stegMode"
                    value="decode"
                    checked={stegMode === 'decode'}
                    onChange={() => setStegMode('decode')}
                  />
                  Decode
                </label>
              </div>
            )}

            {useSteganography && stegMode === 'encode' && (
              <div className="steganography-inputs">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Enter message to hide"
                  className="message-input"
                />
                <input
                  type="text"
                  value={savePath}
                  onChange={(e) => setSavePath(e.target.value)}
                  placeholder="Enter save path (optional)"
                  className="save-path-input"
                />
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={saveToServer}
                    onChange={(e) => setSaveToServer(e.target.checked)}
                  />
                  Save encoded image to server
                </label>
                <div className="button-group">
                  <button
                    type="button"
                    onClick={handleEncode}
                    disabled={!file || !message || loading}
                    className="encode-button"
                  >
                    Encode Message
                  </button>
                </div>
              </div>
            )}

            {useSteganography && stegMode === 'decode' && (
              <div className="steganography-inputs">
                <div className="button-group">
                  <button
                    type="button"
                    onClick={handleDecode}
                    disabled={!file || loading}
                    className="decode-button"
                  >
                    Decode Message
                  </button>
                </div>
              </div>
            )}
          </div>

          {!useSteganography && (
            <button
              type="submit"
              disabled={!file || loading}
              className="submit-button"
            >
              {loading ? 'Processing...' : 'Classify Breed'}
            </button>
          )}
        </form>

        {error && <div className="error-message">{error}</div>}

        {encodedImage && (
          <div className="steganography-result">
            <h2>Encoded Image</h2>
            <img 
              src={`data:image/png;base64,${encodedImage}`} 
              alt="Encoded" 
              className="result-image" 
            />
            <a
              href={`data:image/png;base64,${encodedImage}`}
              download="encoded_image.png"
              className="download-button"
            >
              Download Encoded Image
            </a>
            {serverPath && (
              <div className="server-path">
                <strong>Saved on server at:</strong>
                <div className="path-content">{serverPath}</div>
              </div>
            )}
          </div>
        )}

        {decodedMessage && (
          <div className="decoded-message">
            <h2>Decoded Message</h2>
            <p className="message-content">{decodedMessage}</p>
          </div>
        )}

        {!useSteganography && predictions.length > 0 && (
          <div className="predictions">
            <h2>Top 5 Predictions</h2>
            <ul className="prediction-list">
              {predictions.map((prediction, index) => (
                <li key={index} className="prediction-item">
                  <span className="breed">{prediction.breed}</span>
                  <span className="probability">
                    {(prediction.probability * 100).toFixed(2)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
