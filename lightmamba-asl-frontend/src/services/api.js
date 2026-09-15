import axios from 'axios';

export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 min — video inference can be slow on CPU
});

// Response interceptor — normalize errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.response?.data?.message ||
      error.message ||
      'Unknown error';
    return Promise.reject(new Error(message));
  }
);

export const getHealth = () =>
  apiClient.get('/api/health').then((r) => r.data);

export const getClasses = () =>
  apiClient.get('/api/classes').then((r) => r.data);

export const getModelInfo = () =>
  apiClient.get('/api/model/info').then((r) => r.data);

export const getRootStatus = () =>
  apiClient.get('/').then((r) => r.data);

/**
 * Sends the original MP4 file to the backend for LightMamba-ASL inference.
 * @param {File} file - the original .mp4 File object
 * @returns {Promise<object>} prediction response
 */
export const predictVideo = (file) => {
  const formData = new FormData();
  formData.append('video', file);
  return apiClient
    .post('/api/predict/video', formData)
    .then((r) => r.data);
};

export default apiClient;
