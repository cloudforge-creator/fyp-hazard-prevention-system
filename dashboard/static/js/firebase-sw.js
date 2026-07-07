// firebase-sw.js - Service Worker for background push notifications
// This file must be served at /static/js/firebase-sw.js

importScripts('https://www.gstatic.com/firebasejs/10.7.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.7.0/firebase-messaging-compat.js');

// Same config as register_token.html
firebase.initializeApp({
  apiKey:            "YOUR_API_KEY",
  authDomain:        "YOUR_PROJECT.firebaseapp.com",
  databaseURL:       "https://YOUR_PROJECT-rtdb.firebaseio.com",
  projectId:         "YOUR_PROJECT_ID",
  storageBucket:     "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId:             "YOUR_APP_ID"
});

const messaging = firebase.messaging();

// Handle background messages (when app is not open)
messaging.onBackgroundMessage(function(payload) {
  self.registration.showNotification(
    payload.notification.title,
    {
      body:    payload.notification.body,
      icon:    '/static/icon.png',
      badge:   '/static/badge.png',
      vibrate: [200, 100, 200],    // vibration pattern
      requireInteraction: true      // stay until dismissed
    }
  );
});
