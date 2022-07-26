import React from 'react';

import '@mediapipe/face_mesh';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

let RENDER_PREDICTION_ON_FRAME_HANDLE = -1;
let detector: faceLandmarksDetection.FaceLandmarksDetector;
let canvasContext: CanvasRenderingContext2D | null;

const startStreaming = async (video: HTMLVideoElement, stream: MediaStream): Promise<void> => {
  video.srcObject = stream;
  video.play();
}

const getVideoFeed = async (): Promise<MediaStream | undefined> => {
  try {
  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  return stream;
  } catch (error) {
    console.error('Could not get camera feed!');
    alert((error as Error).message);
  }
}

const performFaceDetection = async (videoFeed: HTMLVideoElement) => {
  if (!detector) {
    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detectorConfig: faceLandmarksDetection.MediaPipeFaceMeshMediaPipeModelConfig = {
      runtime: 'mediapipe', // or 'tfjs'
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
      refineLandmarks: true
    }
    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
  }
  return await detector.estimateFaces(videoFeed);
}

const drawResult = async (originalVideo: HTMLVideoElement, detectionResult: faceLandmarksDetection.Face[]) => {
  console.log('rendering result: ', detectionResult);

  if (!canvasContext) {
    const canvas = document.getElementById('augmentedVideo') as HTMLCanvasElement;
    canvas.width = originalVideo.videoWidth;
    canvas.height = originalVideo.videoHeight;
    canvasContext = canvas.getContext('2d');

    if (!canvasContext) {
      alert('No convas to draw on found');
      return;
    }
  }

  canvasContext.drawImage(originalVideo, 0, 0, originalVideo.videoWidth, originalVideo.videoHeight);

  for (const face of detectionResult) {
    const { keypoints } = face;

    for (const keypoint of keypoints) {
      const {x, y} = keypoint;
      canvasContext.beginPath();
      canvasContext.arc(x, y, 2, 0, 2 * Math.PI);
      canvasContext.fillStyle = 'red';
      canvasContext.fill();
    }
  }
}

const startFaceDetection = async (video: HTMLVideoElement) => {
  console.log('Starting Face Detection on each animation frame');
  RENDER_PREDICTION_ON_FRAME_HANDLE = requestAnimationFrame(async () => {
    const detectionResult = await performFaceDetection(video);
    drawResult(video, detectionResult);
    startFaceDetection(video);
  });
}

const start = async () => {
  console.log('Starting Stream');
  const videoFeed = await getVideoFeed();
  if (!videoFeed) return;

  console.log('Starting Video Feed');
  const originalVideo = document.getElementById('originalVideo') as HTMLVideoElement;
  await startStreaming(originalVideo, videoFeed);

  startFaceDetection(originalVideo);
}

function App() {

  return (
    <div>
      <button onClick={start}>Detect</button>
      <button onClick={
        () => {
          console.log('Stopping Detection');
          cancelAnimationFrame(RENDER_PREDICTION_ON_FRAME_HANDLE);
          RENDER_PREDICTION_ON_FRAME_HANDLE = -1
        }
      }>Stop</button>
      <div>
        <video id='originalVideo' autoPlay />
        <canvas id='augmentedVideo' />
      </div>
    </div>
  );
}

export default App;
