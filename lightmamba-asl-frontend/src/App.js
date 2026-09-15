import React, { useState } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

import Sidebar from './components/Sidebar';
import Navbar  from './components/Navbar';
import Footer  from './components/Footer';

import Home             from './pages/Home';
import VideoRecognition from './pages/VideoRecognition';
import LiveRecognition  from './pages/LiveRecognition';
import SentenceBuilder  from './pages/SentenceBuilder';
import SkeletonDemo     from './pages/SkeletonDemo';
import ModelInfo        from './pages/ModelInfo';
import Results          from './pages/Results';
import About            from './pages/About';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <HashRouter>
      <div className="app-shell">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <div className="main-area">
          <Navbar onMenuToggle={() => setSidebarOpen((o) => !o)} />

          <main className="page-content">
            <Routes>
              <Route path="/"          element={<Home />} />
              <Route path="/recognize" element={<VideoRecognition />} />
              <Route path="/live"      element={<LiveRecognition />} />
              <Route path="/sentence"  element={<SentenceBuilder />} />
              <Route path="/skeleton"  element={<SkeletonDemo />} />
              <Route path="/model"     element={<ModelInfo />} />
              <Route path="/results"   element={<Results />} />
              <Route path="/about"     element={<About />} />
            </Routes>
          </main>

          <Footer />
        </div>
      </div>
    </HashRouter>
  );
}
