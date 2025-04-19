import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Container,
  Typography,
  Paper,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Card,
  CardContent,
  Avatar,
  Fade,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PetsIcon from '@mui/icons-material/Pets';
import axios from 'axios';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#ff4081',
    },
    background: {
      default: '#f5f7fa',
      paper: '#fff',
    },
  },
  typography: {
    fontFamily: 'Poppins, Roboto, Arial, sans-serif',
    h3: {
      fontWeight: 700,
      letterSpacing: '-1px',
    },
    h6: {
      fontWeight: 600,
    },
  },
});

interface Prediction {
  breed: string;
  probability: number;
}

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    setPredictions([]);

    try {
      console.log('Uploading file:', file.name);
      const formData = new FormData();
      formData.append('file', file);

      console.log('Making API request...');
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Raw API response:', response.data);
      
      const preds = response.data?.predictions;
      console.log('Extracted predictions:', preds);

      if (Array.isArray(preds) && preds.length > 0) {
        console.log('Setting predictions:', preds);
        setPredictions(preds);
      } else {
        console.warn('No valid predictions received');
        setPredictions([]);
      }

    } catch (error: any) {
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
      });
      alert(`Error: ${error.response?.data?.detail || error.message || 'Failed to predict dog breed'}`);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    multiple: false,
  });

  const hasPredictions = Array.isArray(predictions) && predictions.length > 0;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Dog Breed Classifier
          </Typography>

          <Paper
            {...getRootProps()}
            sx={{
              p: 4,
              mt: 4,
              mb: 2,
              textAlign: 'center',
              cursor: 'pointer',
              background: isDragActive
                ? 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
                : 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)',
              border: isDragActive ? '2.5px solid #1976d2' : '2.5px dashed #1976d2',
              boxShadow: isDragActive
                ? '0 4px 24px 0 rgba(33,150,243,0.15)'
                : '0 2px 12px 0 rgba(33,150,243,0.07)',
              transition: 'all 0.3s',
              '&:hover': {
                background: 'linear-gradient(135deg, #fcb69f 0%, #ffecd2 100%)',
                border: '2.5px solid #ff4081',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon sx={{ fontSize: 48, color: '#1976d2', mb: 1 }} />
            <Typography variant="h6" sx={{ fontWeight: 500 }}>
              {isDragActive
                ? 'Drop the image here'
                : 'Drag & drop a dog image, or click to select'}
            </Typography>
            <Typography variant="body2" sx={{ color: '#666', mt: 1 }}>
              (JPG, PNG; max 5MB)
            </Typography>
          </Paper>

          {previewUrl && (
            <Fade in={!!previewUrl}>
              <Box sx={{ mt: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Card sx={{ maxWidth: 340, borderRadius: 4, boxShadow: 4 }}>
                  <CardContent sx={{ p: 2 }}>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{
                        width: '100%',
                        maxHeight: '260px',
                        borderRadius: '12px',
                        objectFit: 'cover',
                        boxShadow: '0 2px 12px 0 rgba(33,150,243,0.10)',
                      }}
                    />
                  </CardContent>
                </Card>
              </Box>
            </Fade>
          )}

          {loading && (
            <Box sx={{ mt: 3 }}>
              <CircularProgress />
            </Box>
          )}

          {hasPredictions && (
            <Fade in={hasPredictions}>
              <Paper sx={{ mt: 5, p: 3, borderRadius: 4, boxShadow: 6, background: 'linear-gradient(120deg, #f6d365 0%, #fda085 100%)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <PetsIcon sx={{ color: '#ff4081', fontSize: 32, mr: 1 }} />
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 700, color: '#222' }}>
                    Predictions
                  </Typography>
                </Box>
                <List>
                  {predictions.map((prediction, index) => (
                    <ListItem key={index} sx={{ mb: 1, borderRadius: 2, background: '#fff8', boxShadow: 2 }}>
                      <Avatar sx={{ bgcolor: '#1976d2', mr: 2 }}>
                        {index + 1}
                      </Avatar>
                      <ListItemText
                        primary={<span style={{ fontWeight: 600, fontSize: '1.1em', color: '#222' }}>{prediction.breed.replace('_', ' ')}</span>}
                        secondary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <LinearProgress
                              variant="determinate"
                              value={prediction.probability * 100}
                              sx={{ flexGrow: 1, height: 10, borderRadius: 5, background: '#e3f2fd' }}
                            />
                            <Typography variant="body2" sx={{ minWidth: 50, fontWeight: 500, color: '#1976d2' }}>
                              {(prediction.probability * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Fade>
          )}
        </Box>
        <Box sx={{ mt: 8, mb: 2, textAlign: 'center', opacity: 0.85 }}>
          <Typography variant="caption" sx={{ color: '#aaa' }}>
            &copy; {new Date().getFullYear()} Dog Breed Classifier
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
