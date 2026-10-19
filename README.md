 HEAD
 # SignBridge – Real-time Sign Language to Text and Speech

A browser app that recognises American Sign Language (ASL) fingerspelling from your webcam and turns it into text and speech. Everything runs on your device: no server, no video upload.

Based on the project seminar *"Real-Time Sign Language Detection using Deep Convolutional Neural Networks"* (KVG College of Engineering, Dept. of CS&E).

**Live demo:** `https://<your-github-username>.github.io/<repo-name>/`

## Features

- Live webcam capture with hand-skeleton overlay and a separate skeleton view
- Recognises A–Y by hand shape; **J** and **Z** by tracking the fingertip path; an open hand adds a **space**
- Hold-to-confirm with a progress ring, so noisy frames don't type letters
- Word building with auto-suggestions (click or press 1–4)
- Text-to-speech: speak the sentence, each word on space, or each letter
- **Calibrate tab:** record your own samples to train a k-NN model in the browser; export and commit them so every visitor gets your calibration
- Works on desktop and mobile; light and dark themes; keyboard accessible

## Pipeline (matches the seminar methodology)

Camera → frame capture → hand detection (MediaPipe, 21 landmarks) → feature extraction → classification (rules + trained k-NN) → stabilising → word formation → suggestions → text display and text-to-speech.

| File | Role |
|---|---|
| `js/classifier.js` | Landmark features, rule-based letters, k-NN |
| `js/motion.js` | J and Z from the fingertip path |
| `js/stabilizer.js` | Majority vote and hold timer |
| `js/words.js` | Dictionary and suggestions |
| `js/speech.js` | Web Speech API wrapper |
| `js/main.js` | Camera, UI, calibration |

## Run locally

```bash
npm start            # serves http://localhost:8000 (needs Python 3)
npm test             # unit tests, Node 18+
```
The camera only works on `https://` or `localhost`.

## Deploy on GitHub Pages

1. Create a new public repository (for example `signbridge`) and push this folder to the `main` branch.
2. In the repository: **Settings → Pages → Build and deployment → Source: Deploy from a branch**, branch `main`, folder `/ (root)`, then Save.
3. After a minute the site is live at `https://<username>.github.io/<repo>/`. Put that link on your resume.

## Honest notes on accuracy

- The seminar reports 97% accuracy for the CNN model. **This web version has not been benchmarked on a labelled dataset**, so don't quote that figure for it.
- The rule-based classifier is reliable for letters that differ by finger count (B, U, V, W, I, Y, L, F) and decent for D, O, C, X. Letters that differ only by thumb position (**A, S, E, M, N, T**) and K, P, Q, R are hard for rules. Use the **Calibrate** tab for these.
- Best results: good light, plain background, one hand, palm toward the camera.
- Landmarks come from the MediaPipe Hand Landmarker, loaded from jsDelivr and Google storage on first use, so the first start needs internet.

## Improve accuracy

1. Open the app, go to **Calibrate**, start the camera, and record each letter (about 2.5 seconds each, rotate your hand slightly). Record from different distances for more robustness.
2. **Export samples**, save the file as `data/samples.json`, commit and push.

## Resume bullets (edit to match what you did)

- Built SignBridge, a real-time ASL fingerspelling recognizer (JavaScript, MediaPipe) that converts webcam gestures to text and speech; deployed on GitHub Pages.
- Designed a landmark-feature pipeline with a rule-based classifier and an in-browser trainable k-NN, hold-to-confirm stabilisation, and dictionary auto-suggestions.
- Wrote unit tests for the classifier, motion detector and stabiliser, plus a browser end-to-end test with a simulated camera.

## Licence

MIT. Word frequency data derived from `popular-english-words` (ISC).

# sign-language-bridge
 2c270b0c0eaa9c63f2af188dcdae6549ea9e7e3e
