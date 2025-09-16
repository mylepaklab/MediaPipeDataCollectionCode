import { GestureRecognizer, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";

const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("outputCanvas");
const ctx = canvasElement.getContext("2d");

let gestureRecognizer;

// Start webcam
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  videoElement.srcObject = stream;

  return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      // Set canvas size to video size for clear rendering
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      resolve();
    };
  });
}

// Load model and initialize recognizer
async function createGestureRecognizer() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "./gesture_model.task", // Your zipped model file
      },
      runningMode: "VIDEO",
      maxResults: 1,
    });

    console.log("GestureRecognizer created successfully.");
  } catch (error) {
    console.error("Failed to create GestureRecognizer:", error);
  }
}

// Main loop: run recognition every frame
async function predict() {
  if (!gestureRecognizer) {
    console.warn("GestureRecognizer not initialized yet.");
    return;
  }

  try {
    // Await the async recognition call
    const results = await gestureRecognizer.recognizeForVideo(videoElement, performance.now());

    // Debug output: see full results in console
    console.log("Gesture recognition results:", results);

    // Clear canvas and draw video frame
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    // Check if gestures detected
    if (results.gestures && results.gestures.length > 0) {
      const gesture = results.gestures[0][0]; // top prediction

      // Display gesture and confidence
      ctx.font = "30px Arial";
      ctx.fillStyle = "red";
      ctx.fillText(`Gesture: ${gesture.categoryName} (${(gesture.score * 100).toFixed(1)}%)`, 10, 40);
    }
  } catch (error) {
    console.error("Error during gesture recognition:", error);
  }

  requestAnimationFrame(predict);
}

async function main() {
  await setupCamera();
  videoElement.play();

  await createGestureRecognizer();

  predict();
}

main();

