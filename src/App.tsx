import React, { useEffect } from 'react';

import '@mediapipe/face_mesh';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { VERSION as FaceMeshVersion, FaceMesh, FACEMESH_FACE_OVAL, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_LEFT_IRIS, FACEMESH_LIPS, FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_IRIS, FACEMESH_TESSELATION, Results, NormalizedLandmarkList, LandmarkConnectionArray } from '@mediapipe/face_mesh';

let RENDER_PREDICTION_ON_FRAME_HANDLE = -1;
let canvasContext: CanvasRenderingContext2D | null;
const roseGlasses = new Image();
const flowerCrown = new Image();

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

const drawResult = async (detectionResults: Results) => {
  console.log('rendering result');

  if (!canvasContext) {
    const canvas = document.getElementById('augmentedVideo') as HTMLCanvasElement;
    canvas.width = detectionResults.image.width;
    canvas.height = detectionResults.image.height;
    canvasContext = canvas.getContext('2d');

    if (!canvasContext) {
      alert('No canvas to draw on found');
      return;
    }
  }

  canvasContext.clearRect(0, 0, detectionResults.image.width, detectionResults.image.height);

  const drawBackground = document.getElementById('backgroundToggle') as HTMLInputElement;
  if (drawBackground?.checked) {
    canvasContext.drawImage(detectionResults.image, 0, 0, detectionResults.image.width, detectionResults.image.height);
  }

  const drawOption = (document.getElementById('drawOption') as HTMLSelectElement).value;

  if (drawOption === 'mesh') {
    for (const landmarks of detectionResults.multiFaceLandmarks) {
      drawConnectors(canvasContext, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
      drawConnectors(canvasContext, landmarks, FACEMESH_RIGHT_EYE, {color: '#FF3030'});
      drawConnectors(canvasContext, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030'});
      drawConnectors(canvasContext, landmarks, FACEMESH_RIGHT_IRIS, {color: '#FF3030'});
      drawConnectors(canvasContext, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
      drawConnectors(canvasContext, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#30FF30'});
      drawConnectors(canvasContext, landmarks, FACEMESH_LEFT_IRIS, {color: '#30FF30'});
      drawConnectors(canvasContext, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});
      drawConnectors(canvasContext, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});
    }
  }

  if (drawOption === 'landmarks') {
    for (const landmarks of detectionResults.multiFaceLandmarks) {
      drawLandmarks(canvasContext, landmarks, {color: '#ff0000', radius: 1});
    }
  }

  if (drawOption === 'funFilter') {
    canvasContext.fillStyle = '#B76E7949';
    canvasContext.fillRect(0, 0, detectionResults.image.width, detectionResults.image.height);

    for (const landmarks of detectionResults.multiFaceLandmarks) {
      const rightEye = getTopLeftBottomRight(landmarks, FACEMESH_LEFT_EYE, { width: detectionResults.image.width, height: detectionResults.image.height });
      const leftEye = getTopLeftBottomRight(landmarks, FACEMESH_RIGHT_EYE,  { width: detectionResults.image.width, height: detectionResults.image.height });
      const head = getTopLeftBottomRight(landmarks, FACEMESH_FACE_OVAL, { width: detectionResults.image.width, height: detectionResults.image.height });

      const eyeWidth = Math.abs(rightEye.right - leftEye.left);
      const eyeHeight = Math.abs(leftEye.bottom - leftEye.top);

      const leftPupil = {
        x: leftEye.left + (leftEye.right - leftEye.left) / 2,
        y: leftEye.top + (leftEye.bottom - leftEye.top) / 2,
      };
      const rightPupil = {
        x: rightEye.left + (rightEye.right - rightEye.left) / 2,
        y: rightEye.top + (rightEye.bottom - rightEye.top) / 2,
      };
      const delta = {
        x: rightPupil.x - leftPupil.x,
        y: rightPupil.y - leftPupil.y,
      };
      const rotation = Math.atan2(delta.y, delta.x);

      const glassesWidth = eyeWidth * 1.75;
      const glassesHeight = (head.bottom - head.top) / 3;
      const glassesWidthExcess = glassesWidth - eyeWidth;
      const glassesHeightExcess = glassesHeight - eyeHeight;

      canvasContext.save();
      canvasContext.translate(leftEye.left, leftEye.top);
      canvasContext.rotate(rotation);
      canvasContext.drawImage(
        roseGlasses,
        -glassesWidthExcess / 2,
        -glassesHeightExcess / 2,
        glassesWidth,
        glassesHeight);

      // canvasContext.drawImage(
      //   flowerCrown,
      //   -glassesWidth / 4,
      //   -glassesHeight * 1.75,
      //   glassesWidth * 1.1,
      //   glassesHeight * 1.5
      // );

      canvasContext.restore();
    }
  }
}

const getTopLeftBottomRight = (landmarks: NormalizedLandmarkList, KNOWN_LANDMARK: LandmarkConnectionArray, scale: {width: number; height: number}) => {
  const xVals = KNOWN_LANDMARK.map((index) => landmarks[index[0]].x);
  const yVals = KNOWN_LANDMARK.map((index) => landmarks[index[0]].y);

  return {
    top: Math.min(...yVals) * scale.height,
    left: Math.min(...xVals) * scale.width,
    bottom: Math.max(...yVals) * scale.height,
    right: Math.max(...xVals) * scale.width,
  };
}

const doFaceDetection = async (detector: FaceMesh, video: HTMLVideoElement) => {
  RENDER_PREDICTION_ON_FRAME_HANDLE = requestAnimationFrame(async () => {
    await detector.send({ image: video });
    await doFaceDetection(detector, video);
  });
}

const start = async () => {
  console.log('Starting Stream');
  const videoFeed = await getVideoFeed();
  if (!videoFeed) return;

  console.log('Starting Video Feed');
  const videoElement = document.getElementById('originalVideo') as HTMLVideoElement;
  await startStreaming(videoElement, videoFeed);

  const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@${FaceMeshVersion}/${file}`
  });
  faceMesh.setOptions({
    selfieMode: false,
    enableFaceGeometry: true,
    maxNumFaces: 1,
    refineLandmarks: false
  });
  faceMesh.onResults(async (results) => {
    drawResult(results);
  });
  doFaceDetection(faceMesh, videoElement);
}

function App() {

  useEffect(() => {
    roseGlasses.src = 'rose_glasses.png';
    flowerCrown.src = 'flowercrown.png';
  }, [])

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
      <br />

      <select id="drawOption" defaultValue="funFilter">
        <option value="mesh">Mesh</option>
        <option value="landmarks">Landmarks</option>
        <option value="funFilter">Fun Filter 🎉</option>
        <option value="none">none</option>
      </select>
      <br />

      <input id="backgroundToggle" type="checkbox" defaultChecked={true} />
      <label htmlFor="backgroundToggle">Show Background</label>

      <div>
        <video id='originalVideo' autoPlay />
        <canvas id='augmentedVideo' />
      </div>
    </div>
  );
}

export default App;
